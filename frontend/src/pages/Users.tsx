import { useState, useEffect } from 'react'
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
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '../components/ui/dialog'
import { Progress } from '../components/ui/progress'
import { deleteUser } from '../services/api'
import { getAllUsers, deleteUser as deleteUserSupabase } from '../lib/supabase'

interface User {
  user_id: string
  registered_at: string
  last_auth: string
  total_auths: number
  success_rate: number
  embedding_count: number
  quality_score: number
  status: 'active' | 'inactive'
}

const Users = () => {
  const extractErrorMessage = (error: any, fallback = 'Failed to delete user') => {
    const detail = error?.response?.data?.detail ?? error?.response?.data
    if (typeof detail === 'string') return detail
    if (detail && typeof detail === 'object') {
      if (detail.error) return detail.error
      if (detail.message) return detail.message
      if (detail.detail) return detail.detail
      return JSON.stringify(detail)
    }
    if (error?.message) return error.message
    return fallback
  }

  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  
  // Fetch users from Supabase
  useEffect(() => {
    const fetchUsers = async () => {
      try {
        setLoading(true)
        const supabaseUsers = await getAllUsers()
        // Transform Supabase data to match User interface
        const transformedUsers: User[] = supabaseUsers.map((u: any) => ({
          user_id: u.user_id,
          registered_at: u.created_at || u.registered_at,
          last_auth: u.last_authentication || u.created_at || u.last_auth,
          total_auths: u.total_authentications || 0,
          success_rate: u.success_rate || 0,
          embedding_count: u.embedding_count || 0,
          quality_score: u.quality_score || 0,
          status: u.is_active !== false ? 'active' : 'inactive'
        }))
        setUsers(transformedUsers)
      } catch (error) {
        console.error('Error fetching users:', error)
        toast.error('Failed to fetch users. Using mock data.')
        // Fallback to empty array or show error
        setUsers([])
      } finally {
        setLoading(false)
      }
    }
    fetchUsers()
  }, [])

  const [searchTerm, setSearchTerm] = useState('')
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [userToDelete, setUserToDelete] = useState<User | null>(null)

  const filteredUsers = users.filter(user =>
    user.user_id.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const handleDelete = async (userId: string) => {
    try {
      // Try Supabase first, fallback to API
      try {
        await deleteUserSupabase(userId)
      } catch (supabaseError) {
        // Fallback to API
        await deleteUser(userId)
      }
      setUsers(users.filter(u => u.user_id !== userId))
      toast.success(`User ${userId} deleted successfully`)
      setShowDeleteModal(false)
      setUserToDelete(null)
    } catch (error) {
      console.error('Delete error:', error)
      toast.error(extractErrorMessage(error))
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (loading) {
    return (
      <div className="space-y-6 pb-16">
        <Card variant="elevated" className="p-12 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent mx-auto"></div>
          <p className="text-gray-400 mt-4">Loading users...</p>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6 pb-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div className="flex items-center">
          <div className="w-12 h-12 bg-gradient-blue rounded-xl flex items-center justify-center mr-4">
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

      <Card variant="elevated" className="p-6">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search users by ID..."
            className="w-full pl-12 pr-4 py-3 bg-dark-800 border border-dark-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent transition-all"
          />
        </div>
      </Card>

      <div className="grid lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {filteredUsers.map((user, index) => (
          <motion.div
            key={user.user_id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card variant="elevated" hover glow className="h-full p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center">
                  <div className="w-12 h-12 bg-gradient-blue rounded-full flex items-center justify-center mr-3">
                    <span className="text-white font-bold text-lg">
                      {user.user_id.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white">{user.user_id}</h3>
                    <Badge variant={user.status === 'active' ? 'success' : 'destructive'}>
                      {user.status}
                    </Badge>
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
                  <Badge variant={user.success_rate > 98 ? 'success' : user.success_rate > 95 ? 'warning' : 'destructive'}>
                    {user.success_rate.toFixed(1)}%
                  </Badge>
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
                  <Progress value={user.quality_score * 100} variant="success" />
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      {filteredUsers.length === 0 && (
        <Card variant="elevated" className="text-center py-12">
          <UsersIcon className="w-16 h-16 mx-auto text-gray-600 mb-4" />
          <p className="text-gray-400">No users found matching "{searchTerm}"</p>
        </Card>
      )}

      <Dialog open={!!selectedUser} onOpenChange={() => setSelectedUser(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>User Details</DialogTitle>
            <DialogDescription>
              Detailed information about {selectedUser?.user_id}
            </DialogDescription>
          </DialogHeader>
          {selectedUser && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm text-gray-400">User ID</label>
                  <p className="text-white font-semibold">{selectedUser.user_id}</p>
                </div>
                <div>
                  <label className="text-sm text-gray-400">Status</label>
                  <Badge variant={selectedUser.status === 'active' ? 'success' : 'destructive'}>
                    {selectedUser.status}
                  </Badge>
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
          )}
          <DialogFooter>
            <Button
              variant="destructive"
              icon={<Trash2 className="w-4 h-4" />}
              onClick={() => {
                setUserToDelete(selectedUser)
                setShowDeleteModal(true)
                setSelectedUser(null)
              }}
            >
              Delete User
            </Button>
            <Button variant="secondary" onClick={() => setSelectedUser(null)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showDeleteModal} onOpenChange={setShowDeleteModal}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Deletion</DialogTitle>
            <DialogDescription>
              This action cannot be undone
            </DialogDescription>
          </DialogHeader>
          {userToDelete && (
            <p className="text-gray-300">
              Are you sure you want to delete user <strong className="text-white">{userToDelete.user_id}</strong>? 
              This will permanently remove all associated biometric data.
            </p>
          )}
          <DialogFooter>
            <Button
              variant="destructive"
              icon={<Trash2 className="w-4 h-4" />}
              onClick={() => userToDelete && handleDelete(userToDelete.user_id)}
            >
              Delete
            </Button>
            <Button
              variant="secondary"
              onClick={() => {
                setShowDeleteModal(false)
                setUserToDelete(null)
              }}
            >
              Cancel
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default Users

