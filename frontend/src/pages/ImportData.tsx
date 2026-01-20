import { useState } from 'react'
import Card from '@/components/Card'
import { Upload, Download, Trash2, CheckCircle, AlertCircle } from 'lucide-react'
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const ACCOUNT_ID = 'default'

export default function ImportData() {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<{
    success: boolean
    message: string
    count?: number
  } | null>(null)

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
      // Reset file input
      const fileInput = document.getElementById('file-input') as HTMLInputElement
      if (fileInput) fileInput.value = ''
    } catch (error: any) {
      setResult({
        success: false,
        message: error.response?.data?.detail || 'Failed to upload file',
      })
    } finally {
      setUploading(false)
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
        <p className="text-slate-400">Upload your transaction history via CSV file</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Upload Card */}
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

            {result && (
              <div
                className={`p-4 rounded-lg flex items-start gap-3 ${
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
                    <p className="text-sm text-slate-400 mt-1">
                      {result.count} transactions imported
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Instructions Card */}
        <Card title="CSV Format Instructions">
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium text-white mb-2">Required Columns</h3>
              <ul className="text-sm text-slate-400 space-y-1">
                <li>• <span className="text-slate-300">date</span> - Transaction date (YYYY-MM-DD or MM/DD/YYYY)</li>
                <li>• <span className="text-slate-300">symbol</span> - Stock ticker symbol</li>
                <li>• <span className="text-slate-300">type</span> - Transaction type (buy, sell, dividend)</li>
                <li>• <span className="text-slate-300">quantity</span> - Number of shares</li>
                <li>• <span className="text-slate-300">price</span> - Price per share</li>
                <li>• <span className="text-slate-300">fees</span> - Transaction fees (optional)</li>
                <li>• <span className="text-slate-300">notes</span> - Additional notes (optional)</li>
              </ul>
            </div>

            <div>
              <h3 className="text-sm font-medium text-white mb-2">Transaction Types</h3>
              <ul className="text-sm text-slate-400 space-y-1">
                <li>• <span className="text-slate-300">buy</span> - Purchase of shares</li>
                <li>• <span className="text-slate-300">sell</span> - Sale of shares</li>
                <li>• <span className="text-slate-300">dividend</span> - Dividend payment</li>
                <li>• <span className="text-slate-300">deposit</span> - Cash deposit</li>
                <li>• <span className="text-slate-300">withdrawal</span> - Cash withdrawal</li>
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

      {/* Danger Zone */}
      <Card title="Danger Zone">
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
