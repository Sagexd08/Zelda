import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FileText, Download, Search } from 'lucide-react'
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { EVENT_TYPES, type EventType } from '../utils/constants'
import { getAuthLogs } from '../lib/supabase'

interface LogEntry {
  id: string
  event_type: EventType
  timestamp: string
  user_id?: string
  status: 'success' | 'failure'
  details?: string
}

const Logs = () => {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>([])
  const [selectedFilter, setSelectedFilter] = useState<EventType | 'all'>('all')
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const supabaseLogs = await getAuthLogs(100)
        // Transform Supabase data to match LogEntry interface
        const transformedLogs: LogEntry[] = supabaseLogs.map((log: any) => ({
          id: log.id,
          event_type: log.event_type as EventType,
          timestamp: log.created_at,
          user_id: log.user_id,
          status: log.status as 'success' | 'failure',
          details: typeof log.details === 'string' ? log.details : JSON.stringify(log.details) || log.event_type
        }))
        setLogs(transformedLogs)
        setFilteredLogs(transformedLogs)
      } catch (error) {
        console.error('Error fetching logs:', error)
        // Fallback to empty array
        setLogs([])
        setFilteredLogs([])
      }
    }
    fetchLogs()
  }, [])

  useEffect(() => {
    let filtered = logs

    if (selectedFilter !== 'all') {
      filtered = filtered.filter((log) => log.event_type === selectedFilter)
    }

    if (searchTerm) {
      filtered = filtered.filter(
        (log) =>
          log.user_id?.toLowerCase().includes(searchTerm.toLowerCase()) ||
          log.details?.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    setFilteredLogs(filtered)
  }, [logs, selectedFilter, searchTerm])

  const exportToCSV = () => {
    const headers = ['ID', 'Event Type', 'Timestamp', 'User ID', 'Status', 'Details']
    const rows = filteredLogs.map((log) => [
      log.id,
      log.event_type,
      log.timestamp,
      log.user_id || '',
      log.status,
      log.details || '',
    ])

    const csvContent = [
      headers.join(','),
      ...rows.map((row) => row.map((cell) => `"${cell}"`).join(',')),
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `logs_${new Date().toISOString().split('T')[0]}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div className="flex items-center">
          <div className="w-12 h-12 bg-gradient-blue rounded-xl flex items-center justify-center mr-4">
            <FileText className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">System Logs</h1>
            <p className="text-gray-400 mt-1">Audit trail and system events</p>
          </div>
        </div>
        <Button onClick={exportToCSV} icon={<Download className="w-5 h-5" />}>
          Export CSV
        </Button>
      </motion.div>

      <Card variant="elevated" className="p-6">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search logs..."
              className="w-full pl-12 pr-4 py-3 bg-dark-800 border border-dark-600 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-accent transition-all"
            />
          </div>
          <div className="flex gap-2">
            {(['all', ...Object.values(EVENT_TYPES)] as const).map((filter) => (
              <Button
                key={filter}
                onClick={() => setSelectedFilter(filter)}
                variant={selectedFilter === filter ? 'default' : 'secondary'}
                size="sm"
              >
                {filter.charAt(0).toUpperCase() + filter.slice(1)}
              </Button>
            ))}
          </div>
        </div>
      </Card>

      <Card variant="elevated" className="p-6">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-dark-700">
                <th className="text-left py-4 px-4 text-gray-400 font-semibold">Event Type</th>
                <th className="text-left py-4 px-4 text-gray-400 font-semibold">Timestamp</th>
                <th className="text-left py-4 px-4 text-gray-400 font-semibold">User ID</th>
                <th className="text-left py-4 px-4 text-gray-400 font-semibold">Status</th>
                <th className="text-left py-4 px-4 text-gray-400 font-semibold">Details</th>
              </tr>
            </thead>
            <tbody>
              {filteredLogs.map((log, index) => (
                <motion.tr
                  key={log.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="border-b border-dark-700 hover:bg-dark-800/50 transition-colors"
                >
                  <td className="py-4 px-4">
                    <Badge variant="info">{log.event_type}</Badge>
                  </td>
                  <td className="py-4 px-4 text-gray-300">
                    {new Date(log.timestamp).toLocaleString()}
                  </td>
                  <td className="py-4 px-4 text-gray-300">{log.user_id || 'N/A'}</td>
                  <td className="py-4 px-4">
                    <Badge variant={log.status === 'success' ? 'success' : 'destructive'}>
                      {log.status}
                    </Badge>
                  </td>
                  <td className="py-4 px-4 text-gray-400">{log.details || '-'}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
          {filteredLogs.length === 0 && (
            <div className="text-center py-12 text-gray-400">
              No logs found matching your criteria
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}

export default Logs

