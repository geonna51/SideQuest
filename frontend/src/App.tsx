import { useState } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import BearLogo from './assets/sidequest_bear_logo.png'

type SearchResult = {
  id: string
  title: string
  description: string
  organization: string
  category: string
  location: string
  start_time: string
  end_time: string
  url: string
  source: string
  doc_type: string
  score: number
  reddit_snippet?: string | null
}

function App(): JSX.Element {
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [source, setSource] = useState<string>('all')
  const [results, setResults] = useState<SearchResult[]>([])
  const [answer, setAnswer] = useState<string>('')
  const [answerWarning, setAnswerWarning] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')

  const handleSearch = async (value: string, selectedSource: string = source): Promise<void> => {
    setSearchTerm(value)

    if (value.trim() === '') {
      setResults([])
      setAnswer('')
      setAnswerWarning('')
      setError('')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await fetch(
        `/api/search?q=${encodeURIComponent(value)}&source=${encodeURIComponent(selectedSource)}`
      )

      if (!response.ok) {
        throw new Error(`Search failed with status ${response.status}`)
      }

      const data = await response.json()
      setResults(data.results ?? [])
      setAnswer(data.answer ?? '')
      setAnswerWarning(data.answer_warning ?? '')
    } catch (err) {
      console.error(err)
      setError('Failed to load search results.')
      setResults([])
      setAnswer('')
      setAnswerWarning('')
    } finally {
      setLoading(false)
    }
  }

  const handleSourceChange = async (newSource: string): Promise<void> => {
    setSource(newSource)

    if (searchTerm.trim() !== '') {
      await handleSearch(searchTerm, newSource)
    }
  }

  return (
    <div className="full-body-container">
      <div className="top-text">
        <div className="logo-container">
          <img src={BearLogo} alt="Cornell Bear Quest Logo" style={{ height: '72px', width: 'auto', mixBlendMode: 'multiply' }} />
          <h1 className="sidequest-title">Side<span>Quest</span></h1>
        </div>

        <div
          className="input-box"
          onClick={() => document.getElementById('search-input')?.focus()}
        >
          <img src={SearchIcon} alt="search" />
          <input
            id="search-input"
            placeholder="Search for things to do in Ithaca..."
            value={searchTerm}
            onChange={(e) => void handleSearch(e.target.value)}
          />
        </div>

        <div style={{ marginTop: '12px' }}>
          <select
            value={source}
            onChange={(e) => void handleSourceChange(e.target.value)}
          >
            <option value="all">All categories</option>
            <option value="events">Events & Activities</option>
            <option value="places">Interesting Places</option>
            <option value="food">Food & Dining</option>
            <option value="outdoors">Outdoors & Trails</option>
            <option value="fitness">Fitness & Rec</option>
          </select>
        </div>
      </div>

      <div id="answer-box">
        {loading && <p className="status-message loading-pulse">Searching the area...</p>}
        {error && <p className="status-message error-message">{error}</p>}

        {!loading && !error && answer && (
          <div className="episode-item">
            <h3 className="episode-title">Synthesized answer</h3>
            <p className="episode-desc">{answer}</p>
          </div>
        )}

        {!loading && !error && answerWarning && (
          <p>{answerWarning}</p>
        )}

        {!loading && !error && results.length === 0 && searchTerm.trim() !== '' && (
          <p className="status-message empty-state">We couldn't find any activities matching your quest.</p>
        )}

        {results.map((result) => (
          <div key={result.id} className="episode-item">
            <h3 className="episode-title">{result.title}</h3>

            {result.description && (
              <p className="episode-desc">{result.description}</p>
            )}

            {result.reddit_snippet && (
              <div style={{ padding: '12px 16px', marginTop: '1rem', marginBottom: '1rem', backgroundColor: '#f9fafb', borderLeft: '3px solid #d1d5db', fontStyle: 'italic', fontSize: '0.95em', color: '#4b5563', borderRadius: '0 8px 8px 0' }}>
                💡 {result.reddit_snippet}
              </div>
            )}

            <div className="episode-meta-container">
              {result.source && (
                <span className="meta-chip source-chip">
                  {result.source}
                </span>
              )}
              {result.category && (
                <span className="meta-chip">
                  {result.category}
                </span>
              )}
              {result.location && (
                <span className="meta-chip">
                  📍 {result.location}
                </span>
              )}
              {result.start_time && (
                <span className="meta-chip">
                  🕒 {result.start_time}
                </span>
              )}
              {result.organization && (
                <span className="meta-chip">
                  Host: {result.organization}
                </span>
              )}
            </div>

            <div className="episode-actions">
              <span className="meta-chip score-chip">Match: {Math.round(result.score * 100)}%</span>
              
              {result.url && result.source !== 'osm' && (
                <a href={result.url} target="_blank" rel="noreferrer" className="action-button">
                  View Details
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default App
