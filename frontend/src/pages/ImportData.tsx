import { useState, useEffect } from 'react'
import Card from '@/components/Card'
import { Upload, Download, Trash2, CheckCircle, AlertCircle, Plus, List } from 'lucide-react'
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const ACCOUNT_ID = 'default'

interface Transaction {
  id: number
  date: string
  symbol: string
  transaction_type: string
  quantity: number
  price: number
  fees: number
  amount: number
  notes?: string
}

export default function ImportData() {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<{
    success: boolean
    message: string
    count?: number
  } | null>(null)

  // Manual entry form state
  const [formData, setFormData] = useState({
    date: new Date().toISOString().split('T')[0],
    symbol: '',
    type: 'buy',
    quantity: '',
    price: '',
    fees: '0',
    notes: '',
  })
  const [symbolValidation, setSymbolValidation] = useState<{
    checking: boolean
    valid: boolean
    name?: string
    error?: string
  }>({ checking: false, valid: false })
  const [submitting, setSubmitting] = useState(false)

  // Transaction list state
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [loadingTransactions, setLoadingTransactions] = useState(false)
  const [showTransactions, setShowTransactions] = useState(false)

  useEffect(() => {
    if (showTransactions) {
      loadTransactions()
    }
  }, [showTransactions])

  const loadTransactions = async () => {
    setLoadingTransactions(true)
    try {
      const response = await axios.get(`${API_BASE_URL}/api/transaction-list/${ACCOUNT_ID}`)
      setTransactions(response.data.transactions)
    } catch (error) {
      console.error('Error loading transactions:', error)
    } finally {
      setLoadingTransactions(false)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setResult(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(
        `${API_BASE_URL}/api/upload-csv/${ACCOUNT_ID}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      )

      setResult({
        success: true,
        message: response.data.message,
        count: response.data.transactions_imported,
      })

      setFile(null)
      const fileInput = document.getElementById('file-input') as HTMLInputElement
      if (fileInput) fileInput.value = ''

      if (showTransactions) {
        loadTransactions()
      }
    } catch (error: any) {
      setResult({
        success: false,
        message: error.response?.data?.detail || 'Failed to upload file',
      })
    } finally {
      setUploading(false)
    }
  }

  const validateSymbol = async (symbol: string) => {
    if (!symbol || symbol.length === 0) {
      setSymbolValidation({ checking: false, valid: false })
      return
    }

    setSymbolValidation({ checking: true, valid: false })

    try {
      const response = await axios.get(`${API_BASE_URL}/api/validate-ticker/${symbol}`)
      if (response.data.valid) {
        setSymbolValidation({
          checking: false,
          valid: true,
          name: response.data.name,
        })
      } else {
        setSymbolValidation({
          checking: false,
          valid: false,
          error: response.data.error,
        })
      }
    } catch (error) {
      setSymbolValidation({
        checking: false,
        valid: false,
        error: 'Error validating ticker',
      })
    }
  }

  const handleSymbolChange = (value: string) => {
    const upperValue = value.toUpperCase()
    setFormData({ ...formData, symbol: upperValue })

    // Debounce validation
    const timeoutId = setTimeout(() => validateSymbol(upperValue), 500)
    return () => clearTimeout(timeoutId)
  }

  const handleSubmitTransaction = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!symbolValidation.valid && formData.type !== 'deposit' && formData.type !== 'withdrawal') {
      setResult({
        success: false,
        message: 'Please enter a valid ticker symbol',
      })
      return
    }

    setSubmitting(true)
    setResult(null)

    try {
      const response = await axios.post(`${API_BASE_URL}/api/transaction/${ACCOUNT_ID}`, formData)

      setResult({
        success: true,
        message: response.data.message,
      })

      // Reset form
      setFormData({
        date: new Date().toISOString().split('T')[0],
        symbol: '',
        type: 'buy',
        quantity: '',
        price: '',
        fees: '0',
        notes: '',
      })
      setSymbolValidation({ checking: false, valid: false })

      if (showTransactions) {
        loadTransactions()
      }
    } catch (error: any) {
      setResult({
        success: false,
        message: error.response?.data?.detail || 'Failed to add transaction',
      })
    } finally {
      setSubmitting(false)
    }
  }

  const handleDownloadTemplate = () => {
    window.location.href = `${API_BASE_URL}/api/download-template`
  }

  const handleDeleteAll = async () => {
    if (!confirm('Are you sure you want to delete all transactions? This cannot be undone.')) {
      return
    }

    try {
      const response = await axios.delete(`${API_BASE_URL}/api/transactions/${ACCOUNT_ID}`)
      setResult({
        success: true,
        message: response.data.message,
      })
      setTransactions([])
    } catch (error: any) {
      setResult({
        success: false,
        message: error.response?.data?.detail || 'Failed to delete transactions',
      })
    }
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Import Portfolio Data</h1>
        <p className="text-slate-400">Upload CSV or manually enter your transaction history</p>
      </div>

      {/* Manual Entry Form */}
      <Card title="Add Transaction Manually" className="mb-6">
        <form onSubmit={handleSubmitTransaction} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Date *</label>
              <input
                type="date"
                value={formData.date}
                onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Symbol *</label>
              <input
                type="text"
                value={formData.symbol}
                onChange={(e) => handleSymbolChange(e.target.value)}
                placeholder="AAPL"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                required={formData.type !== 'deposit' && formData.type !== 'withdrawal'}
                disabled={formData.type === 'deposit' || formData.type === 'withdrawal'}
              />
              {symbolValidation.checking && (
                <p className="text-xs text-slate-400 mt-1">Validating ticker...</p>
              )}
              {symbolValidation.valid && (
                <p className="text-xs text-green-400 mt-1 flex items-center gap-1">
                  <CheckCircle size={12} />
                  {symbolValidation.name}
                </p>
              )}
              {symbolValidation.error && (
                <p className="text-xs text-red-400 mt-1 flex items-center gap-1">
                  <AlertCircle size={12} />
                  {symbolValidation.error}
                </p>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Type *</label>
              <select
                value={formData.type}
                onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
                <option value="dividend">Dividend</option>
                <option value="deposit">Deposit (Cash)</option>
                <option value="withdrawal">Withdrawal (Cash)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Quantity *</label>
              <input
                type="number"
                step="0.0001"
                value={formData.quantity}
                onChange={(e) => setFormData({ ...formData, quantity: e.target.value })}
                placeholder="100"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Price *</label>
              <input
                type="number"
                step="0.01"
                value={formData.price}
                onChange={(e) => setFormData({ ...formData, price: e.target.value })}
                placeholder="150.00"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Fees</label>
              <input
                type="number"
                step="0.01"
                value={formData.fees}
                onChange={(e) => setFormData({ ...formData, fees: e.target.value })}
                placeholder="9.99"
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Notes</label>
            <input
              type="text"
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              placeholder="Optional notes"
              className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-slate-700 disabled:text-slate-400 disabled:cursor-not-allowed transition-colors"
          >
            <Plus size={20} />
            {submitting ? 'Adding...' : 'Add Transaction'}
          </button>
        </form>

        {result && (
          <div
            className={`mt-4 p-4 rounded-lg flex items-start gap-3 ${
              result.success
                ? 'bg-green-500/10 border border-green-500/30'
                : 'bg-red-500/10 border border-red-500/30'
            }`}
          >
            {result.success ? (
              <CheckCircle size={20} className="text-green-400 flex-shrink-0 mt-0.5" />
            ) : (
              <AlertCircle size={20} className="text-red-400 flex-shrink-0 mt-0.5" />
            )}
            <div>
              <p
                className={`text-sm font-medium ${
                  result.success ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {result.message}
              </p>
              {result.count && (
                <p className="text-sm text-slate-400 mt-1">{result.count} transactions imported</p>
              )}
            </div>
          </div>
        )}
      </Card>

      {/* CSV Upload and Instructions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card title="Upload Transactions CSV">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Select CSV File
              </label>
              <input
                id="file-input"
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="block w-full text-sm text-slate-300
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-semibold
                  file:bg-primary-600 file:text-white
                  hover:file:bg-primary-700
                  cursor-pointer"
              />
              {file && (
                <p className="mt-2 text-sm text-slate-400">
                  Selected: {file.name} ({(file.size / 1024).toFixed(2)} KB)
                </p>
              )}
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-slate-700 disabled:text-slate-400 disabled:cursor-not-allowed transition-colors"
            >
              <Upload size={20} />
              {uploading ? 'Uploading...' : 'Upload Transactions'}
            </button>
          </div>
        </Card>

        <Card title="CSV Format Instructions">
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium text-white mb-2">Required Columns</h3>
              <ul className="text-sm text-slate-400 space-y-1">
                <li>
                  • <span className="text-slate-300">date</span> - Transaction date (YYYY-MM-DD)
                </li>
                <li>
                  • <span className="text-slate-300">symbol</span> - Valid stock ticker
                </li>
                <li>
                  • <span className="text-slate-300">type</span> - buy, sell, dividend, deposit,
                  withdrawal
                </li>
                <li>
                  • <span className="text-slate-300">quantity</span> - Number of shares
                </li>
                <li>
                  • <span className="text-slate-300">price</span> - Price per share
                </li>
                <li>
                  • <span className="text-slate-300">fees</span> - Transaction fees (optional)
                </li>
                <li>
                  • <span className="text-slate-300">notes</span> - Notes (optional)
                </li>
              </ul>
            </div>

            <button
              onClick={handleDownloadTemplate}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
            >
              <Download size={20} />
              Download Template CSV
            </button>
          </div>
        </Card>
      </div>

      {/* Transaction List */}
      <Card
        title="Transaction History"
        action={
          <button
            onClick={() => {
              setShowTransactions(!showTransactions)
              if (!showTransactions) loadTransactions()
            }}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
          >
            <List size={16} />
            {showTransactions ? 'Hide' : 'Show'} Transactions
          </button>
        }
      >
        {showTransactions && (
          <div className="space-y-4">
            {loadingTransactions ? (
              <p className="text-slate-400 text-center py-4">Loading transactions...</p>
            ) : transactions.length === 0 ? (
              <p className="text-slate-400 text-center py-4">No transactions found</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left py-2 px-2 text-slate-400 font-medium">Date</th>
                      <th className="text-left py-2 px-2 text-slate-400 font-medium">Symbol</th>
                      <th className="text-left py-2 px-2 text-slate-400 font-medium">Type</th>
                      <th className="text-right py-2 px-2 text-slate-400 font-medium">Quantity</th>
                      <th className="text-right py-2 px-2 text-slate-400 font-medium">Price</th>
                      <th className="text-right py-2 px-2 text-slate-400 font-medium">Amount</th>
                    </tr>
                  </thead>
                  <tbody>
                    {transactions.map((txn) => (
                      <tr key={txn.id} className="border-b border-slate-700/50">
                        <td className="py-2 px-2 text-slate-300">
                          {new Date(txn.date).toLocaleDateString()}
                        </td>
                        <td className="py-2 px-2 text-white font-medium">{txn.symbol}</td>
                        <td className="py-2 px-2">
                          <span
                            className={`px-2 py-0.5 rounded text-xs font-medium ${
                              txn.transaction_type === 'buy'
                                ? 'bg-green-500/20 text-green-400'
                                : txn.transaction_type === 'sell'
                                ? 'bg-red-500/20 text-red-400'
                                : 'bg-blue-500/20 text-blue-400'
                            }`}
                          >
                            {txn.transaction_type}
                          </span>
                        </td>
                        <td className="py-2 px-2 text-right text-slate-300">
                          {txn.quantity.toFixed(4)}
                        </td>
                        <td className="py-2 px-2 text-right text-slate-300">
                          ${txn.price.toFixed(2)}
                        </td>
                        <td
                          className={`py-2 px-2 text-right font-medium ${
                            txn.amount >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}
                        >
                          ${Math.abs(txn.amount).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Danger Zone */}
      <Card title="Danger Zone" className="mt-8">
        <div className="space-y-4">
          <p className="text-sm text-slate-400">
            Delete all imported transactions. This action cannot be undone.
          </p>
          <button
            onClick={handleDeleteAll}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            <Trash2 size={18} />
            Delete All Transactions
          </button>
        </div>
      </Card>
    </div>
  )
}
