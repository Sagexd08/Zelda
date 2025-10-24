import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Users as UsersIcon, 
  Search, 
  Trash2, 
  Eye,
  Calendar,
  Shield,
  AlertTriangle
} from 'lucide-react'
import { toast } from 'sonner'
import Card from '../components/Card'
import Button from '../components/Button'
import { deleteUser } from '../api/client'

const Users = () => {
  // Mock user data - in production this would come from an API
  const [users, setUsers] = useState([
    {
      user_id: 'john_doe',
      registered_at: '2024-01-15T10:30:00',
      last_auth: '2024-01-20T14:22:00',
      total_auths: 45,
      success_rate: 98.5,
      embedding_count: 5,
      quality_score: 0.95,
      status: 'active'
    },
    {
      user_id: 'jane_smith',
      registered_at: '2024-01-18T09:15:00',
      last_auth: '2024-01-21T11:45:00',
      total_auths: 32,
      success_rate: 99.2,
      embedding_count: 4,
      quality_score: 0.97,
      status: 'active'
    },
    {
      user_id: 'bob_wilson',
      registered_at: '2024-01-10T16:20:00',
      last_auth: '2024-01-19T09:30:00',
      total_auths: 78,
      success_rate: 97.8,
      embedding_count: 6,
      quality_score: 0.92,
      status: 'active'
    },
  ])

  const [searchTerm, setSearchTerm] = useState('')
  const [selectedUser, setSelectedUser] = useState(null)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [userToDelete, setUserToDelete] = useState(null)

  const filteredUsers = users.filter(user =>
    user.user_id.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const handleDelete = async (userId) => {
    try {
      await deleteUser(userId)
      setUsers(users.filter(u => u.user_id !== userId))
      toast.success(`User ${userId} deleted successfully`)
      setShowDeleteModal(false)
      setUserToDelete(null)
    } catch (error) {
      console.error('Delete error:', error)
      toast.error(error.response?.data?.detail || 'Failed to delete user')
    }
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="space-y-6 pb-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div className="flex items-center">
          <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center mr-4">
            <UsersIcon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">User Management</h1>
            <p className="text-gray-400 mt-1">Manage registered users and their data</p>
          </div>
        </div>
        <div className="glass px-4 py-2 rounded-lg">
          <span className="text-gray-400">Total Users: </span>
          <span className="text-white font-bold text-xl">{users.length}</span>
        </div>
      </motion.div>

      {/* Search Bar */}
      <Card>
        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search users by ID..."
            className="w-full pl-12 pr-4 py-3 bg-dark-800 border border-dark-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 transition-all"
          />
        </div>
      </Card>

      {/* Users Grid */}
      <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {filteredUsers.map((user, index) => (
          <motion.div
            key={user.user_id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="h-full">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center">
                  <div className="w-12 h-12 bg-gradient-to-br from-primary-500 to-purple-600 rounded-full flex items-center justify-center mr-3">
                    <span className="text-white font-bold text-lg">
                      {user.user_id.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white">{user.user_id}</h3>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      user.status === 'active' 
                        ? 'bg-green-500/20 text-green-500' 
                        : 'bg-red-500/20 text-red-500'
                    }`}>
                      {user.status}
                    </span>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setSelectedUser(user)}
                    className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                  >
                    <Eye className="w-4 h-4 text-gray-400" />
                  </button>
                  <button
                    onClick={() => {
                      setUserToDelete(user)
                      setShowDeleteModal(true)
                    }}
                    className="p-2 hover:bg-red-500/10 rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4 text-red-500" />
                  </button>
                </div>
              </div>

              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400 flex items-center">
                    <Calendar className="w-4 h-4 mr-1" />
                    Registered
                  </span>
                  <span className="text-white">{formatDate(user.registered_at)}</span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Last Auth</span>
                  <span className="text-white">{formatDate(user.last_auth)}</span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Total Auths</span>
                  <span className="text-white font-semibold">{user.total_auths}</span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Success Rate</span>
                  <span className={`font-semibold ${
                    user.success_rate > 98 ? 'text-green-500' : 
                    user.success_rate > 95 ? 'text-yellow-500' : 
                    'text-red-500'
                  }`}>
                    {user.success_rate.toFixed(1)}%
                  </span>
                </div>

                <div className="flex items-center justify-between pt-3 border-t border-dark-700">
                  <span className="text-gray-400 flex items-center">
                    <Shield className="w-4 h-4 mr-1" />
                    Embeddings
                  </span>
                  <span className="text-white font-semibold">{user.embedding_count}</span>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">Quality Score</span>
                    <span className="text-white">{(user.quality_score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-primary-500 to-purple-600"
                      style={{ width: `${user.quality_score * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      {filteredUsers.length === 0 && (
        <Card className="text-center py-12">
          <UsersIcon className="w-16 h-16 mx-auto text-gray-600 mb-4" />
          <p className="text-gray-400">No users found matching "{searchTerm}"</p>
        </Card>
      )}

      {/* User Detail Modal */}
      <AnimatePresence>
        {selectedUser && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedUser(null)}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="glass rounded-xl p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">User Details</h2>
                <button
                  onClick={() => setSelectedUser(null)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-gray-400">User ID</label>
                    <p className="text-white font-semibold">{selectedUser.user_id}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Status</label>
                    <p className="text-white font-semibold capitalize">{selectedUser.status}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Registered At</label>
                    <p className="text-white font-semibold">{formatDate(selectedUser.registered_at)}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Last Authentication</label>
                    <p className="text-white font-semibold">{formatDate(selectedUser.last_auth)}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Total Authentications</label>
                    <p className="text-white font-semibold">{selectedUser.total_auths}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Success Rate</label>
                    <p className="text-white font-semibold">{selectedUser.success_rate.toFixed(2)}%</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Embedding Count</label>
                    <p className="text-white font-semibold">{selectedUser.embedding_count}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Quality Score</label>
                    <p className="text-white font-semibold">{(selectedUser.quality_score * 100).toFixed(1)}%</p>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex space-x-3">
                <Button
                  variant="danger"
                  icon={Trash2}
                  onClick={() => {
                    setUserToDelete(selectedUser)
                    setShowDeleteModal(true)
                    setSelectedUser(null)
                  }}
                  className="flex-1"
                >
                  Delete User
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => setSelectedUser(null)}
                  className="flex-1"
                >
                  Close
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Delete Confirmation Modal */}
      <AnimatePresence>
        {showDeleteModal && userToDelete && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowDeleteModal(false)}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="glass rounded-xl p-6 max-w-md w-full border-red-500/50"
            >
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-red-500/20 rounded-full flex items-center justify-center mr-3">
                  <AlertTriangle className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">Confirm Deletion</h2>
                  <p className="text-gray-400 text-sm">This action cannot be undone</p>
                </div>
              </div>

              <p className="text-gray-300 mb-6">
                Are you sure you want to delete user <strong className="text-white">{userToDelete.user_id}</strong>? 
                This will permanently remove all associated biometric data.
              </p>

              <div className="flex space-x-3">
                <Button
                  variant="danger"
                  icon={Trash2}
                  onClick={() => handleDelete(userToDelete.user_id)}
                  className="flex-1"
                >
                  Delete
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    setShowDeleteModal(false)
                    setUserToDelete(null)
                  }}
                  className="flex-1"
                >
                  Cancel
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default Users


