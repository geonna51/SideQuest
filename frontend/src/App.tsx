import { useState, useEffect } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import BearLogo from './assets/sidequest_bear_logo.png'

const TIME_OPTIONS = [
  { value: 'any', label: 'Any time', queryTerms: '' },
  { value: 'today', label: 'Today', queryTerms: 'today happening now current event open today' },
  { value: 'week', label: 'This week', queryTerms: 'this week upcoming soon weekly event this week' },
  { value: 'weekend', label: 'Weekend', queryTerms: 'weekend saturday sunday weekend activity' },
  { value: 'late', label: 'Late night', queryTerms: 'late night evening after dark open late' },
] as const

const AREA_OPTIONS = [
  { value: 'any', label: 'Anywhere', queryTerms: '' },
  { value: 'campus', label: 'On campus', queryTerms: 'cornell campus collegetown central campus north campus west campus' },
  { value: 'collegetown', label: 'Collegetown', queryTerms: 'collegetown near campus college avenue dryden road' },
  { value: 'downtown', label: 'Downtown Ithaca', queryTerms: 'downtown ithaca commons city center' },
  { value: 'nature', label: 'Nature spots', queryTerms: 'state park trail gorge waterfall nature ithaca outskirts' },
] as const

const INTENT_OPTIONS = [
  { value: 'any', label: 'Any vibe', queryTerms: '' },
  { value: 'study', label: 'Quiet study', queryTerms: 'quiet study focus laptop reading calm coffee' },
  { value: 'social', label: 'Meet people', queryTerms: 'social group meet people community conversation club' },
  { value: 'food', label: 'Cheap food', queryTerms: 'cheap food affordable quick bite casual' },
  { value: 'active', label: 'Get active', queryTerms: 'active workout exercise hiking fitness movement' },
  { value: 'relax', label: 'Relaxing', queryTerms: 'relax peaceful cozy low key chill' },
] as const

type SearchContextOption = {
  value: string
  label: string
  queryTerms: string
}

type LatentDimension = {
  dimension: number
  direction: 'positive' | 'negative'
  weight?: number
  query_weight?: number
  document_weight?: number
  alignment?: number
  positive_terms: string[]
  negative_terms: string[]
}

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
  search_mode: 'svd' | 'tfidf'
  matched_dimensions?: LatentDimension[]
}

function App(): JSX.Element {
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [source, setSource] = useState<string>('all')
  const [searchMode, setSearchMode] = useState<'svd' | 'tfidf'>('svd')
  const [timeFilter, setTimeFilter] = useState<string>('any')
  const [areaFilter, setAreaFilter] = useState<string>('any')
  const [intentFilter, setIntentFilter] = useState<string>('any')
  const [futureOnly, setFutureOnly] = useState<boolean>(true)
  const [dateFrom, setDateFrom] = useState<string>('')
  const [dateTo, setDateTo] = useState<string>('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [visibleCount, setVisibleCount] = useState<number>(10)
  const [answer, setAnswer] = useState<string>('')
  const [answerWarning, setAnswerWarning] = useState<string>('')
  const [queryLatentProfile, setQueryLatentProfile] = useState<{ positive: LatentDimension[]; negative: LatentDimension[] }>({ positive: [], negative: [] })
  const [effectiveMode, setEffectiveMode] = useState<'svd' | 'tfidf'>('svd')
  const [retrievalContext, setRetrievalContext] = useState<string[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [sortBy, setSortBy] = useState<'score' | 'date_asc' | 'date_desc'>('score')
  const hasSynthesisAnswer = answer.trim() !== ''
  const hasSynthesisWarning = answerWarning.trim() !== ''

  // Mirrors backend parse_event_date: extracts "20 January 2026" from freeform strings
  const parseStartTime = (str: string): number => {
    if (!str) return NaN
    const iso = Date.parse(str)
    if (!isNaN(iso)) return iso
    const match = str.match(/(\d{1,2})\s+(\w+)\s+(\d{4})/)
    if (match) return Date.parse(`${match[1]} ${match[2]} ${match[3]}`)
    return NaN
  }

  const sortedResults = [...results].sort((a, b) => {
    if (sortBy === 'score') return 0
    const aTime = parseStartTime(a.start_time)
    const bTime = parseStartTime(b.start_time)
    const aValid = !isNaN(aTime)
    const bValid = !isNaN(bTime)
    if (!aValid && !bValid) return 0
    if (!aValid) return 1  // undated results go last
    if (!bValid) return -1
    return sortBy === 'date_asc' ? aTime - bTime : bTime - aTime
  })

  // Read state from URL on first mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const q = params.get('q') ?? ''
    const src = params.get('source') ?? 'all'
    const mode = (params.get('mode') ?? 'svd') as 'svd' | 'tfidf'
    const time = params.get('time') ?? 'any'
    const area = params.get('area') ?? 'any'
    const intent = params.get('intent') ?? 'any'
    const future = params.get('future_only') !== 'false'
    const from = params.get('date_from') ?? ''
    const to = params.get('date_to') ?? ''

    setSource(src)
    setSearchMode(mode)
    setTimeFilter(time)
    setAreaFilter(area)
    setIntentFilter(intent)
    setFutureOnly(future)
    setDateFrom(from)
    setDateTo(to)

    if (q) void handleSearch(q, src, mode, time, area, intent, future, from, to)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Sync URL whenever search state changes
  useEffect(() => {
    const params = new URLSearchParams()
    if (searchTerm) params.set('q', searchTerm)
    if (source !== 'all') params.set('source', source)
    if (searchMode !== 'svd') params.set('mode', searchMode)
    if (timeFilter !== 'any') params.set('time', timeFilter)
    if (areaFilter !== 'any') params.set('area', areaFilter)
    if (intentFilter !== 'any') params.set('intent', intentFilter)
    if (!futureOnly) params.set('future_only', 'false')
    if (dateFrom) params.set('date_from', dateFrom)
    if (dateTo) params.set('date_to', dateTo)
    const qs = params.toString()
    window.history.replaceState(null, '', qs ? `?${qs}` : window.location.pathname)
  }, [searchTerm, source, searchMode, timeFilter, areaFilter, intentFilter, futureOnly, dateFrom, dateTo])

  const buildAugmentedQuery = (
    baseQuery: string,
    selectedTime: string = timeFilter,
    selectedArea: string = areaFilter,
    selectedIntent: string = intentFilter
  ): { composedQuery: string; labels: string[] } => {
    const selections = [
      TIME_OPTIONS.find((option) => option.value === selectedTime),
      AREA_OPTIONS.find((option) => option.value === selectedArea),
      INTENT_OPTIONS.find((option) => option.value === selectedIntent),
    ].filter(Boolean) as SearchContextOption[]

    const contextTerms = selections
      .map((option) => option.queryTerms.trim())
      .filter((terms) => terms.length > 0)

    const labels = selections
      .filter((option) => option.value !== 'any')
      .map((option) => option.label)

    return {
      composedQuery: [baseQuery.trim(), ...contextTerms].filter(Boolean).join(' ').trim(),
      labels,
    }
  }

  const handleSearch = async (
    value: string,
    selectedSource: string = source,
    selectedMode: 'svd' | 'tfidf' = searchMode,
    selectedTime: string = timeFilter,
    selectedArea: string = areaFilter,
    selectedIntent: string = intentFilter,
    selectedFutureOnly: boolean = futureOnly,
    selectedDateFrom: string = dateFrom,
    selectedDateTo: string = dateTo
  ): Promise<void> => {
    setSearchTerm(value)

    if (value.trim() === '') {
      setResults([])
      setVisibleCount(10)
      setAnswer('')
      setAnswerWarning('')
      setQueryLatentProfile({ positive: [], negative: [] })
      setRetrievalContext([])
      setError('')
      return
    }

    const { composedQuery, labels } = buildAugmentedQuery(value, selectedTime, selectedArea, selectedIntent)
    setLoading(true)
    setVisibleCount(10)
    setError('')
    setRetrievalContext(labels)

    try {
      let apiUrl = `/api/search?q=${encodeURIComponent(composedQuery)}&source=${encodeURIComponent(selectedSource)}&mode=${encodeURIComponent(selectedMode)}&future_only=${selectedFutureOnly}&top_k=30`
      if (selectedDateFrom) apiUrl += `&date_from=${encodeURIComponent(selectedDateFrom)}`
      if (selectedDateTo) apiUrl += `&date_to=${encodeURIComponent(selectedDateTo)}`
      const response = await fetch(apiUrl)

      if (!response.ok) {
        throw new Error(`Search failed with status ${response.status}`)
      }

      const data = await response.json()
      setResults(data.results ?? [])
      setAnswer(data.answer ?? '')
      setAnswerWarning(data.answer_warning ?? '')
      setQueryLatentProfile(data.query_latent_profile ?? { positive: [], negative: [] })
      setEffectiveMode(data.effective_mode ?? selectedMode)
    } catch (err) {
      console.error(err)
      setError('Failed to load search results.')
      setResults([])
      setAnswer('')
      setAnswerWarning('')
      setQueryLatentProfile({ positive: [], negative: [] })
      setRetrievalContext(labels)
    } finally {
      setLoading(false)
    }
  }

  const handleSourceChange = async (newSource: string): Promise<void> => {
    setSource(newSource)

    if (searchTerm.trim() !== '') {
      await handleSearch(searchTerm, newSource, searchMode, timeFilter, areaFilter, intentFilter)
    }
  }

  const handleModeChange = async (newMode: 'svd' | 'tfidf'): Promise<void> => {
    setSearchMode(newMode)

    if (searchTerm.trim() !== '') {
      await handleSearch(searchTerm, source, newMode, timeFilter, areaFilter, intentFilter)
    }
  }

  const handleContextChange = async (
    nextTime: string = timeFilter,
    nextArea: string = areaFilter,
    nextIntent: string = intentFilter
  ): Promise<void> => {
    if (searchTerm.trim() !== '') {
      await handleSearch(searchTerm, source, searchMode, nextTime, nextArea, nextIntent)
    } else {
      const { labels } = buildAugmentedQuery('', nextTime, nextArea, nextIntent)
      setRetrievalContext(labels)
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

        <section className="search-controls-card" aria-label="Search filters">
          <div className="filter-control">
            <label htmlFor="source-filter" className="filter-label">Category</label>
            <select
              id="source-filter"
              className="filter-select"
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

          <div className="filter-grid">
            <div className="filter-control">
              <label htmlFor="time-filter" className="filter-label">When</label>
              <select
                id="time-filter"
                className="filter-select"
                value={timeFilter}
                onChange={(e) => {
                  const next = e.target.value
                  setTimeFilter(next)
                  void handleContextChange(next, areaFilter, intentFilter)
                }}
              >
                {TIME_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </div>

            <div className="filter-control">
              <label htmlFor="area-filter" className="filter-label">Area</label>
              <select
                id="area-filter"
                className="filter-select"
                value={areaFilter}
                onChange={(e) => {
                  const next = e.target.value
                  setAreaFilter(next)
                  void handleContextChange(timeFilter, next, intentFilter)
                }}
              >
                {AREA_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </div>

            <div className="filter-control">
              <label htmlFor="intent-filter" className="filter-label">Intent</label>
              <select
                id="intent-filter"
                className="filter-select"
                value={intentFilter}
                onChange={(e) => {
                  const next = e.target.value
                  setIntentFilter(next)
                  void handleContextChange(timeFilter, areaFilter, next)
                }}
              >
                {INTENT_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="filter-toggle-row">
            <label className="filter-toggle-label">
              <input
                type="checkbox"
                checked={futureOnly}
                onChange={(e) => {
                  const next = e.target.checked
                  setFutureOnly(next)
                  if (searchTerm.trim() !== '') {
                    void handleSearch(searchTerm, source, searchMode, timeFilter, areaFilter, intentFilter, next)
                  }
                }}
              />
              Upcoming events only
            </label>
          </div>

          <div className="filter-date-row">
            <div className="filter-control">
              <label htmlFor="date-from" className="filter-label">From</label>
              <input
                id="date-from"
                type="date"
                className="filter-select"
                value={dateFrom}
                onChange={(e) => {
                  const next = e.target.value
                  setDateFrom(next)
                  if (searchTerm.trim() !== '') {
                    void handleSearch(searchTerm, source, searchMode, timeFilter, areaFilter, intentFilter, futureOnly, next, dateTo)
                  }
                }}
              />
            </div>
            <div className="filter-control">
              <label htmlFor="date-to" className="filter-label">To</label>
              <input
                id="date-to"
                type="date"
                className="filter-select"
                value={dateTo}
                onChange={(e) => {
                  const next = e.target.value
                  setDateTo(next)
                  if (searchTerm.trim() !== '') {
                    void handleSearch(searchTerm, source, searchMode, timeFilter, areaFilter, intentFilter, futureOnly, dateFrom, next)
                  }
                }}
              />
            </div>
            {(dateFrom || dateTo) && (
              <button
                type="button"
                className="date-clear-button"
                onClick={() => {
                  setDateFrom('')
                  setDateTo('')
                  if (searchTerm.trim() !== '') {
                    void handleSearch(searchTerm, source, searchMode, timeFilter, areaFilter, intentFilter, futureOnly, '', '')
                  }
                }}
              >
                Clear dates
              </button>
            )}
          </div>

          <fieldset className="filter-mode-group">
            <legend className="filter-label">Ranking mode</legend>
            <div className="mode-toggle-row" role="radiogroup" aria-label="Search ranking mode">
              <button
                type="button"
                className={`mode-toggle-button ${searchMode === 'svd' ? 'active' : ''}`}
                aria-pressed={searchMode === 'svd'}
                onClick={() => void handleModeChange('svd')}
              >
                <span className="mode-toggle-title">SVD Search</span>
                <span className="mode-toggle-subtitle">Latent semantic ranking</span>
              </button>
              <button
                type="button"
                className={`mode-toggle-button ${searchMode === 'tfidf' ? 'active' : ''}`}
                aria-pressed={searchMode === 'tfidf'}
                onClick={() => void handleModeChange('tfidf')}
              >
                <span className="mode-toggle-title">TF-IDF Baseline</span>
                <span className="mode-toggle-subtitle">Exact lexical matching</span>
              </button>
            </div>
          </fieldset>
        </section>
      </div>

      <div id="answer-box">
        {loading && (
          <div className="loading-state">
            <div className="spinner" aria-hidden="true" />
            <p className="status-message">Searching the area...</p>
          </div>
        )}
        {error && <p className="status-message error-message">{error}</p>}

        {!loading && !error && searchTerm.trim() !== '' && (
          <section className={`synthesis-card ${hasSynthesisWarning && !hasSynthesisAnswer ? 'synthesis-card-warning' : ''}`} aria-live="polite">
            <div className="synthesis-header">
              <div>
                <p className="synthesis-eyebrow">LLM Synthesis</p>
                <h2 className="synthesis-title">Quick recommendation summary</h2>
              </div>
              <span className={`synthesis-status-pill ${hasSynthesisAnswer ? 'ready' : 'offline'}`}>
                {hasSynthesisAnswer ? 'Available' : 'Unavailable'}
              </span>
            </div>

            {hasSynthesisAnswer && (
              <p className="episode-desc synthesis-copy">{answer}</p>
            )}

            {!hasSynthesisAnswer && hasSynthesisWarning && (
              <div className="synthesis-empty-state">
                <p className="synthesis-empty-title">Search results are still available.</p>
                <p className="synthesis-empty-copy">
                  The app found matching activities, but the summary assistant is currently offline because the backend API key is not configured.
                </p>
                <p className="synthesis-empty-hint">{answerWarning}</p>
              </div>
            )}

            {!hasSynthesisAnswer && !hasSynthesisWarning && (
              <div className="synthesis-empty-state">
                <p className="synthesis-empty-title">A summary will appear here after results load.</p>
                <p className="synthesis-empty-copy">
                  Use this panel to compare how the ranked results roll up into a short recommendation.
                </p>
              </div>
            )}

            {retrievalContext.length > 0 && (
              <div className="context-chip-bar" aria-label="Active retrieval context">
                {retrievalContext.map((label) => (
                  <span key={label} className="context-chip">{label}</span>
                ))}
              </div>
            )}
          </section>
        )}

        {!loading && !error && searchTerm.trim() !== '' && (
          <div className="mode-summary-card">
            <p className="mode-summary-label">
              Search mode in use: <strong>{effectiveMode === 'svd' ? 'SVD latent retrieval' : 'TF-IDF lexical retrieval'}</strong>
            </p>
            {effectiveMode === 'svd' && (
              <p className="mode-summary-note">Community snippets from Reddit are matched separately by keyword overlap and shown alongside results.</p>
            )}
            {effectiveMode === 'svd' && (
              <div className="dimension-groups">
                <div>
                  <p className="dimension-group-title">Top positive query dimensions</p>
                  <div className="dimension-chip-row">
                    {queryLatentProfile.positive.length > 0 ? queryLatentProfile.positive.map((dimension) => (
                      <span key={`positive-${dimension.dimension}`} className="dimension-chip">
                        D{dimension.dimension} (+): {dimension.positive_terms.slice(0, 3).join(', ')}
                      </span>
                    )) : <span className="dimension-chip muted-chip">No strong positive dimensions</span>}
                  </div>
                </div>

                <div>
                  <p className="dimension-group-title">Top negative query dimensions</p>
                  <div className="dimension-chip-row">
                    {queryLatentProfile.negative.length > 0 ? queryLatentProfile.negative.map((dimension) => (
                      <span key={`negative-${dimension.dimension}`} className="dimension-chip negative-chip">
                        D{dimension.dimension} (-): {dimension.negative_terms.slice(0, 3).join(', ')}
                      </span>
                    )) : <span className="dimension-chip muted-chip">No strong negative dimensions</span>}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {!loading && !error && results.length === 0 && searchTerm.trim() !== '' && (
          <p className="status-message empty-state">We couldn't find any activities matching your quest.</p>
        )}

        {results.length > 0 && (
          <div className="sort-row">
            <label htmlFor="sort-select" className="filter-label">Sort by</label>
            <select
              id="sort-select"
              className="sort-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'score' | 'date_asc' | 'date_desc')}
            >
              <option value="score">Best match</option>
              <option value="date_asc">Date: soonest first</option>
              <option value="date_desc">Date: latest first</option>
            </select>
          </div>
        )}

        {sortedResults.slice(0, visibleCount).map((result) => (
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

            {result.search_mode === 'svd' && result.matched_dimensions && result.matched_dimensions.length > 0 && (
              <div className="dimension-panel">
                <p className="dimension-group-title">Latent dimensions matched</p>
                <div className="dimension-chip-row">
                  {result.matched_dimensions.map((dimension) => (
                    <span
                      key={`${result.id}-${dimension.dimension}-${dimension.direction}`}
                      className={`dimension-chip ${dimension.direction === 'negative' ? 'negative-chip' : ''}`}
                    >
                      D{dimension.dimension} ({dimension.direction === 'positive' ? '+' : '-'}): {dimension.direction === 'positive'
                        ? dimension.positive_terms.slice(0, 3).join(', ')
                        : dimension.negative_terms.slice(0, 3).join(', ')}
                    </span>
                  ))}
                </div>
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

        {!loading && visibleCount < sortedResults.length && (
          <div className="show-more-row">
            <button
              type="button"
              className="show-more-button"
              onClick={() => setVisibleCount((c) => c + 10)}
            >
              Show more ({sortedResults.length - visibleCount} remaining)
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
