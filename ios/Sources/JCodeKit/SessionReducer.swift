import Foundation

/// One entry in the rendered transcript.
public struct TranscriptEntry: Equatable, Sendable, Identifiable {
    public enum Role: String, Sendable {
        case user
        case assistant
        case system
    }

    public struct ToolCall: Equatable, Sendable, Identifiable {
        public enum Status: Equatable, Sendable {
            case streamingInput
            case running
            case succeeded
            case failed(String)
        }

        public var id: String
        public var name: String
        public var input: String
        public var output: String
        public var status: Status

        public init(
            id: String, name: String, input: String = "", output: String = "",
            status: Status = .streamingInput
        ) {
            self.id = id
            self.name = name
            self.input = input
            self.output = output
            self.status = status
        }
    }

    public var id: UUID
    public var role: Role
    public var text: String
    public var reasoning: String
    public var toolCalls: [ToolCall]
    /// True while this entry is still receiving streamed content.
    public var isStreaming: Bool

    public init(
        id: UUID = UUID(),
        role: Role,
        text: String,
        reasoning: String = "",
        toolCalls: [ToolCall] = [],
        isStreaming: Bool = false
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.reasoning = reasoning
        self.toolCalls = toolCalls
        self.isStreaming = isStreaming
    }
}

/// Full client-side session state derived from server events.
public struct SessionState: Equatable, Sendable {
    public var phase: ConnectionPhase
    public var transcript: [TranscriptEntry]
    public var sessionID: String?
    public var sessionTitle: String?
    public var allSessions: [String]
    public var isProcessing: Bool
    public var isReasoning: Bool
    public var modelName: String?
    public var providerName: String?
    public var availableModels: [String]
    public var serverVersion: String?
    public var tokenInput: UInt64
    public var tokenOutput: UInt64
    public var statusDetail: String?
    public var errorBanner: String?
    public var notices: [String]

    public init() {
        phase = .disconnected
        transcript = []
        sessionID = nil
        sessionTitle = nil
        allSessions = []
        isProcessing = false
        isReasoning = false
        modelName = nil
        providerName = nil
        availableModels = []
        serverVersion = nil
        tokenInput = 0
        tokenOutput = 0
        statusDetail = nil
        errorBanner = nil
        notices = []
    }
}

/// Local intents that mutate state without a server round-trip.
public enum LocalIntent: Equatable, Sendable {
    /// User submitted a message; append it optimistically.
    case userSentMessage(String)
    /// User queued a soft-interrupt message mid-run.
    case userQueuedInterrupt(String)
    /// Dismiss the current error banner.
    case dismissError
    /// Reset everything (switching servers/sessions).
    case reset
}

/// Pure state machine turning connection output into session state.
///
/// All streaming, tool lifecycle, and history-sync behavior lives here so it
/// can be exhaustively unit tested on macOS without UI or network.
public enum SessionReducer {
    public static func reduce(_ state: SessionState, _ output: ConnectionOutput) -> SessionState {
        switch output {
        case .phase(let phase):
            return reducePhase(state, phase)
        case .event(let event):
            return reduceEvent(state, event)
        }
    }

    public static func reduce(_ state: SessionState, intent: LocalIntent) -> SessionState {
        var state = state
        switch intent {
        case .userSentMessage(let text):
            state.transcript.append(TranscriptEntry(role: .user, text: text))
            state.isProcessing = true
            state.errorBanner = nil
        case .userQueuedInterrupt(let text):
            state.transcript.append(TranscriptEntry(role: .user, text: text))
        case .dismissError:
            state.errorBanner = nil
        case .reset:
            state = SessionState()
        }
        return state
    }

    // MARK: - Phase

    private static func reducePhase(_ state: SessionState, _ phase: ConnectionPhase)
        -> SessionState
    {
        var state = state
        state.phase = phase
        switch phase {
        case .connected:
            state.errorBanner = nil
        case .failed(let reason):
            state.errorBanner = reason
            state.isProcessing = false
            state.isReasoning = false
        case .disconnected, .reconnecting:
            state.isProcessing = false
            state.isReasoning = false
            finishStreaming(&state)
        case .connecting:
            break
        }
        return state
    }

    // MARK: - Events

    private static func reduceEvent(_ state: SessionState, _ event: ServerEvent) -> SessionState {
        var state = state
        switch event {
        case .ack:
            state.isProcessing = true

        case .textDelta(let text):
            withStreamingAssistant(&state) { $0.text += text }

        case .reasoningDelta(let text):
            state.isReasoning = true
            withStreamingAssistant(&state) { $0.reasoning += text }

        case .reasoningDone:
            state.isReasoning = false

        case .textReplace(let text):
            withStreamingAssistant(&state) { $0.text = text }

        case .toolStart(let id, let name):
            withStreamingAssistant(&state) { entry in
                entry.toolCalls.append(.init(id: id, name: name))
            }

        case .toolInput(let delta):
            withStreamingAssistant(&state) { entry in
                if !entry.toolCalls.isEmpty {
                    entry.toolCalls[entry.toolCalls.count - 1].input += delta
                }
            }

        case .toolExec(let id, let name):
            withStreamingAssistant(&state) { entry in
                if let index = entry.toolCalls.firstIndex(where: { $0.id == id }) {
                    entry.toolCalls[index].status = .running
                } else {
                    entry.toolCalls.append(.init(id: id, name: name, status: .running))
                }
            }

        case .toolDone(let id, let name, let output, let error):
            withStreamingAssistant(&state) { entry in
                if let index = entry.toolCalls.firstIndex(where: { $0.id == id }) {
                    entry.toolCalls[index].output = output
                    entry.toolCalls[index].status = error.map { .failed($0) } ?? .succeeded
                } else {
                    entry.toolCalls.append(
                        .init(
                            id: id, name: name, output: output,
                            status: error.map { .failed($0) } ?? .succeeded
                        ))
                }
            }

        case .messageEnd:
            finishStreaming(&state)

        case .done:
            state.isProcessing = false
            state.isReasoning = false
            finishStreaming(&state)

        case .interrupted:
            state.isProcessing = false
            state.isReasoning = false
            finishStreaming(&state)
            state.notices.append("Interrupted")

        case .error(_, let message, let retryAfterSecs):
            state.isProcessing = false
            state.isReasoning = false
            finishStreaming(&state)
            if let retry = retryAfterSecs {
                state.errorBanner = "\(message) (retry in \(retry)s)"
            } else {
                state.errorBanner = message
            }

        case .tokenUsage(let input, let output):
            state.tokenInput = input
            state.tokenOutput = output

        case .statusDetail(let detail):
            state.statusDetail = detail

        case .state(_, let sessionID, _, let isProcessing):
            state.sessionID = sessionID
            state.isProcessing = isProcessing

        case .sessionID(let sessionID):
            state.sessionID = sessionID

        case .sessionRenamed(let sessionID, let displayTitle):
            if state.sessionID == nil || state.sessionID == sessionID {
                state.sessionTitle = displayTitle
            }

        case .history(let payload):
            state = applyHistory(state, payload)

        case .modelChanged(_, let model, let error):
            if let error {
                state.errorBanner = error
            } else {
                state.modelName = model
            }

        case .availableModelsUpdated(let models, let providerModel):
            state.availableModels = models
            if let providerModel {
                state.modelName = providerModel
            }

        case .compaction(let trigger, let tokensSaved):
            if let saved = tokensSaved, trigger != "background" {
                state.notices.append("Context compacted (\(saved) tokens saved)")
            }

        case .notification(let fromName, let message):
            let prefix = fromName.map { "\($0): " } ?? ""
            state.notices.append(prefix + message)

        case .pong, .unknown:
            break
        }
        return state
    }

    private static func applyHistory(
        _ state: SessionState, _ payload: ServerEvent.HistoryPayload
    ) -> SessionState {
        var state = state
        state.sessionID = payload.sessionID
        state.providerName = payload.providerName ?? state.providerName
        state.modelName = payload.providerModel ?? state.modelName
        if !payload.availableModels.isEmpty {
            state.availableModels = payload.availableModels
        }
        if !payload.allSessions.isEmpty {
            state.allSessions = payload.allSessions
        }
        state.serverVersion = payload.serverVersion ?? state.serverVersion
        state.sessionTitle = payload.displayTitle ?? state.sessionTitle
        if let totals = payload.totalTokens {
            state.tokenInput = totals.input
            state.tokenOutput = totals.output
        }

        // History replaces the transcript wholesale: it is the server's
        // authoritative view, used on connect and reconnect.
        state.transcript = payload.messages.compactMap { message in
            let role: TranscriptEntry.Role
            switch message.role {
            case "user": role = .user
            case "assistant": role = .assistant
            case "system": role = .system
            default: return nil
            }
            var toolCalls: [TranscriptEntry.ToolCall] = []
            if let data = message.toolData {
                toolCalls.append(
                    .init(
                        id: data.id,
                        name: data.name,
                        input: data.input,
                        output: data.output ?? "",
                        status: data.error.map { .failed($0) }
                            ?? (data.output != nil ? .succeeded : .running)
                    ))
            } else {
                toolCalls = message.toolCalls.map { name in
                    .init(id: name, name: name, status: .succeeded)
                }
            }
            // Skip empty assistant placeholders.
            if message.content.isEmpty && toolCalls.isEmpty {
                return nil
            }
            return TranscriptEntry(role: role, text: message.content, toolCalls: toolCalls)
        }
        return state
    }

    // MARK: - Helpers

    /// Mutates the trailing streaming assistant entry, creating it if needed.
    private static func withStreamingAssistant(
        _ state: inout SessionState, _ mutate: (inout TranscriptEntry) -> Void
    ) {
        if let last = state.transcript.indices.last,
            state.transcript[last].role == .assistant,
            state.transcript[last].isStreaming
        {
            mutate(&state.transcript[last])
        } else {
            var entry = TranscriptEntry(role: .assistant, text: "", isStreaming: true)
            mutate(&entry)
            state.transcript.append(entry)
        }
    }

    private static func finishStreaming(_ state: inout SessionState) {
        if let last = state.transcript.indices.last, state.transcript[last].isStreaming {
            state.transcript[last].isStreaming = false
            // Drop fully-empty assistant stubs (e.g. tool-only turns that
            // were replaced or cancelled before any text arrived).
            if state.transcript[last].text.isEmpty
                && state.transcript[last].toolCalls.isEmpty
                && state.transcript[last].reasoning.isEmpty
            {
                state.transcript.removeLast()
            }
        }
    }
}
