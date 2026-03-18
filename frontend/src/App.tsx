import { useState } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'

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
}

function App(): JSX.Element {
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [source, setSource] = useState<string>('all')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')

  const handleSearch = async (value: string, selectedSource: string = source): Promise<void> => {
    setSearchTerm(value)

    if (value.trim() === '') {
      setResults([])
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
    } catch (err) {
      console.error(err)
      setError('Failed to load search results.')
      setResults([])
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
        <h1 className="sidequest-title">SideQuest</h1>

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
            <option value="all">All sources</option>
            <option value="campusgroups">CampusGroups</option>
            <option value="reddit">Reddit</option>
          </select>
        </div>
      </div>

      <div id="answer-box">
        {loading && <p>Loading...</p>}
        {error && <p>{error}</p>}

        {!loading && !error && results.length === 0 && searchTerm.trim() !== '' && (
          <p>No results found.</p>
        )}

        {results.map((result) => (
          <div key={result.id} className="episode-item">
            <h3 className="episode-title">{result.title}</h3>

            {result.description && (
              <p className="episode-desc">{result.description}</p>
            )}

            <p className="episode-rating">
              Source: {result.source}
            </p>

            {result.organization && (
              <p className="episode-rating">
                Organization: {result.organization}
              </p>
            )}

            {result.category && (
              <p className="episode-rating">
                Category: {result.category}
              </p>
            )}

            {result.location && (
              <p className="episode-rating">
                Location: {result.location}
              </p>
            )}

            {result.start_time && (
              <p className="episode-rating">
                Start: {result.start_time}
              </p>
            )}

            <p className="episode-rating">
              Similarity Score: {result.score}
            </p>

            {result.url && (
              <p>
                <a href={result.url} target="_blank" rel="noreferrer">
                  Open link
                </a>
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default App