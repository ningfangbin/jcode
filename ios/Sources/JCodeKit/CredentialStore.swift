import Foundation

/// A saved server credential.
public struct ServerCredential: Codable, Equatable, Sendable, Identifiable {
    public var id: String { "\(host):\(port)" }

    public var host: String
    public var port: UInt16
    public var token: String
    public var serverName: String
    public var serverVersion: String
    public var pairedAt: Date

    public init(
        host: String,
        port: UInt16,
        token: String,
        serverName: String,
        serverVersion: String,
        pairedAt: Date = Date()
    ) {
        self.host = host
        self.port = port
        self.token = token
        self.serverName = serverName
        self.serverVersion = serverVersion
        self.pairedAt = pairedAt
    }

    public var gateway: Gateway {
        Gateway(host: host, port: port)
    }
}

/// Storage for paired-server credentials.
public protocol CredentialStore: Sendable {
    func loadAll() -> [ServerCredential]
    func save(_ credential: ServerCredential)
    func remove(id: String)
}

/// Keychain-backed store used on device. All credentials live under one
/// keychain item as a JSON array; the auth token is the only secret and the
/// set is expected to stay tiny.
public struct KeychainCredentialStore: CredentialStore {
    private static let service = "com.jcode.mobile.servers"
    private static let account = "paired-servers"

    public init() {}

    public func loadAll() -> [ServerCredential] {
        var query = baseQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return [] }
        return (try? JSONDecoder().decode([ServerCredential].self, from: data)) ?? []
    }

    public func save(_ credential: ServerCredential) {
        var all = loadAll().filter { $0.id != credential.id }
        all.append(credential)
        persist(all)
    }

    public func remove(id: String) {
        persist(loadAll().filter { $0.id != id })
    }

    private func persist(_ credentials: [ServerCredential]) {
        guard let data = try? JSONEncoder().encode(credentials) else { return }
        var query = baseQuery()
        let attributes: [String: Any] = [kSecValueData as String: data]
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            query[kSecValueData as String] = data
            query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
            SecItemAdd(query as CFDictionary, nil)
        }
    }

    private func baseQuery() -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.service,
            kSecAttrAccount as String: Self.account,
        ]
    }
}

/// In-memory store for tests and previews.
public final class InMemoryCredentialStore: CredentialStore, @unchecked Sendable {
    private let lock = NSLock()
    private var credentials: [ServerCredential] = []

    public init() {}

    public func loadAll() -> [ServerCredential] {
        lock.withLock { credentials }
    }

    public func save(_ credential: ServerCredential) {
        lock.withLock {
            credentials.removeAll { $0.id == credential.id }
            credentials.append(credential)
        }
    }

    public func remove(id: String) {
        lock.withLock {
            credentials.removeAll { $0.id == id }
        }
    }
}
