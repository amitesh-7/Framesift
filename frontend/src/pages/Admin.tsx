import { useState, useEffect } from "react";
import { FloatingNav, BackgroundPathsWrapper, Button } from "@/components/ui";
import { Footer } from "@/components/layout";
import { Shield, User, Mail, Key, Calendar, LogIn } from "lucide-react";
import { config } from "@/config";

interface UserLogin {
  id: string;
  name: string;
  email: string;
  picture: string;
  lastLogin: string;
  loginCount: number;
}

const navItems = [
  { name: "Home", link: "/" },
  { name: "Features", link: "/features" },
  { name: "Search", link: "/search" },
];

export function AdminPage() {
  const [adminEmail, setAdminEmail] = useState("");
  const [adminKey, setAdminKey] = useState("");
  const [isAdminAuthenticated, setIsAdminAuthenticated] = useState(false);
  const [authError, setAuthError] = useState("");

  const [users, setUsers] = useState<UserLogin[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isAdminAuthenticated) {
      fetchUsers();
      // Poll for updates every 10 seconds
      const interval = setInterval(fetchUsers, 10000);
      return () => clearInterval(interval);
    }
  }, [isAdminAuthenticated]);

  const handleAdminLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError("");

    // Check if email is in admin list and key matches
    if (
      config.adminEmails.includes(adminEmail) &&
      adminKey === config.adminKey
    ) {
      setIsAdminAuthenticated(true);
      setAuthError("");
    } else {
      setAuthError("Invalid email or admin key");
    }
  };

  const fetchUsers = async () => {
    try {
      const response = await fetch(`${config.apiUrl}/admin/users`, {
        headers: {
          "X-Admin-Key": adminKey,
        },
      });

      if (!response.ok) {
        throw new Error("Failed to fetch users");
      }

      const data = await response.json();
      setUsers(data.users || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load users");
    } finally {
      setLoading(false);
    }
  };

  // Show login form if not authenticated
  if (!isAdminAuthenticated) {
    return (
      <div className="min-h-screen bg-black">
        <FloatingNav navItems={navItems} />
        <BackgroundPathsWrapper>
          <div className="min-h-screen flex items-center justify-center pt-20 px-4">
            <div className="w-full max-w-md">
              <div className="border border-zinc-800 rounded-2xl p-8 bg-zinc-900/50 backdrop-blur-sm">
                <div className="text-center mb-8">
                  <Shield className="w-16 h-16 text-violet-500 mx-auto mb-4" />
                  <h1 className="text-2xl font-bold text-white mb-2">
                    Admin Portal
                  </h1>
                  <p className="text-zinc-400">
                    Enter your credentials to continue
                  </p>
                </div>

                <form onSubmit={handleAdminLogin} className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-2">
                      Admin Email
                    </label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-500" />
                      <input
                        type="email"
                        value={adminEmail}
                        onChange={(e) => setAdminEmail(e.target.value)}
                        placeholder="admin@example.com"
                        className="w-full pl-10 pr-4 py-3 bg-zinc-800/50 border border-zinc-700 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                        required
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-2">
                      Admin Key
                    </label>
                    <div className="relative">
                      <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-500" />
                      <input
                        type="password"
                        value={adminKey}
                        onChange={(e) => setAdminKey(e.target.value)}
                        placeholder="Enter admin key"
                        className="w-full pl-10 pr-4 py-3 bg-zinc-800/50 border border-zinc-700 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                        required
                      />
                    </div>
                  </div>

                  {authError && (
                    <div className="border border-red-500/20 bg-red-500/10 rounded-lg p-3">
                      <p className="text-red-400 text-sm">{authError}</p>
                    </div>
                  )}

                  <Button type="submit" className="w-full">
                    <LogIn className="w-4 h-4 mr-2" />
                    Sign In to Admin Portal
                  </Button>
                </form>
              </div>
            </div>
          </div>
          <Footer />
        </BackgroundPathsWrapper>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black">
      <FloatingNav navItems={navItems} />

      <BackgroundPathsWrapper>
        <div className="max-w-7xl mx-auto px-4 pt-28 pb-20">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-2">
              <Shield className="w-8 h-8 text-violet-500" />
              <h1 className="text-3xl font-bold text-white">Admin Portal</h1>
            </div>
            <p className="text-zinc-400">
              Monitor user logins and authentication activity
            </p>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="border border-zinc-800 rounded-2xl p-6 bg-zinc-900/50 backdrop-blur-sm">
              <div className="flex items-center gap-3 mb-2">
                <User className="w-5 h-5 text-violet-400" />
                <p className="text-zinc-400 text-sm">Total Users</p>
              </div>
              <p className="text-3xl font-bold text-white">{users.length}</p>
            </div>

            <div className="border border-zinc-800 rounded-2xl p-6 bg-zinc-900/50 backdrop-blur-sm">
              <div className="flex items-center gap-3 mb-2">
                <Calendar className="w-5 h-5 text-fuchsia-400" />
                <p className="text-zinc-400 text-sm">Total Logins</p>
              </div>
              <p className="text-3xl font-bold text-white">
                {users.reduce((acc, u) => acc + u.loginCount, 0)}
              </p>
            </div>

            <div className="border border-zinc-800 rounded-2xl p-6 bg-zinc-900/50 backdrop-blur-sm">
              <div className="flex items-center gap-3 mb-2">
                <Shield className="w-5 h-5 text-emerald-400" />
                <p className="text-zinc-400 text-sm">Active Sessions</p>
              </div>
              <p className="text-3xl font-bold text-white">{users.length}</p>
            </div>
          </div>

          {/* Error State */}
          {error && (
            <div className="border border-red-500/20 bg-red-500/10 rounded-2xl p-4 mb-6">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {/* Loading State */}
          {loading ? (
            <div className="text-center py-12">
              <div className="inline-block w-8 h-8 border-4 border-violet-500/30 border-t-violet-500 rounded-full animate-spin"></div>
              <p className="text-zinc-400 mt-4">Loading user data...</p>
            </div>
          ) : (
            /* Users Table */
            <div className="border border-zinc-800 rounded-2xl overflow-hidden bg-zinc-900/50 backdrop-blur-sm">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-zinc-800">
                      <th className="text-left p-4 text-sm font-semibold text-zinc-400">
                        User
                      </th>
                      <th className="text-left p-4 text-sm font-semibold text-zinc-400">
                        Email
                      </th>
                      <th className="text-left p-4 text-sm font-semibold text-zinc-400">
                        Google ID
                      </th>
                      <th className="text-left p-4 text-sm font-semibold text-zinc-400">
                        Last Login
                      </th>
                      <th className="text-left p-4 text-sm font-semibold text-zinc-400">
                        Login Count
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="text-center p-12">
                          <User className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                          <p className="text-zinc-500">
                            No users logged in yet
                          </p>
                        </td>
                      </tr>
                    ) : (
                      users.map((userLogin) => (
                        <tr
                          key={userLogin.id}
                          className="border-b border-zinc-800 hover:bg-zinc-800/50 transition-colors"
                        >
                          <td className="p-4">
                            <div className="flex items-center gap-3">
                              <img
                                src={userLogin.picture}
                                alt={userLogin.name}
                                className="w-10 h-10 rounded-full"
                              />
                              <div>
                                <p className="text-white font-medium">
                                  {userLogin.name}
                                </p>
                              </div>
                            </div>
                          </td>
                          <td className="p-4">
                            <div className="flex items-center gap-2 text-zinc-400">
                              <Mail className="w-4 h-4" />
                              <span className="text-sm">{userLogin.email}</span>
                            </div>
                          </td>
                          <td className="p-4">
                            <div className="flex items-center gap-2 text-zinc-400">
                              <Key className="w-4 h-4" />
                              <span className="text-sm font-mono">
                                {userLogin.id.substring(0, 16)}...
                              </span>
                            </div>
                          </td>
                          <td className="p-4">
                            <span className="text-sm text-zinc-400">
                              {new Date(userLogin.lastLogin).toLocaleString()}
                            </span>
                          </td>
                          <td className="p-4">
                            <span className="inline-flex items-center px-2.5 py-1 rounded-full bg-violet-500/20 text-violet-400 text-sm font-medium">
                              {userLogin.loginCount}
                            </span>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
        <Footer />
      </BackgroundPathsWrapper>
    </div>
  );
}
