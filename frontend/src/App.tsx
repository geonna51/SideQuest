import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'
import SearchIcon from './assets/mag.png'
import BearLogo from './assets/sidequest_bear_logo.png'

type AppPage = 'search' | 'about'

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

type PlacesData = {
  website?: string | null
  phone?: string | null
  hours?: string[] | null
  rating?: number | null
  rating_count?: number | null
  price_level?: string | null
  photo_url?: string | null
  reviews?: Array<{
    author: string
    rating: number | null
    text: string
    relative_time: string
  }> | null
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
  lat?: number | null
  lon?: number | null
  places_data?: PlacesData | null
  reddit_snippet?: string | null
  search_mode: 'svd' | 'tfidf'
  matched_dimensions?: LatentDimension[]
}

const getCurrentPage = (): AppPage => {
  const hash = window.location.hash.replace(/^#/, '')
  return hash === '/about' ? 'about' : 'search'
}

function App(): JSX.Element {
  const [page, setPage] = useState<AppPage>(getCurrentPage)
  const [searchInput, setSearchInput] = useState<string>('')
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [source, setSource] = useState<string>('all')
  const [searchMode, setSearchMode] = useState<'svd' | 'tfidf'>('svd')
  const [timeFilter, setTimeFilter] = useState<string>('any')
  const [areaFilter, setAreaFilter] = useState<string>('any')
  const [intentFilter, setIntentFilter] = useState<string>('any')
  const [futureOnly, setFutureOnly] = useState<boolean>(true)
  const [includeReddit, setIncludeReddit] = useState<boolean>(true)
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
  const [summaryLoading, setSummaryLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [sortBy, setSortBy] = useState<'score' | 'date_asc' | 'date_desc'>('score')
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null)
  const [rewrittenQuery, setRewrittenQuery] = useState<string | null>(null)
  const [chatOpen, setChatOpen] = useState(false)
  const [chatMessages, setChatMessages] = useState<Array<{text: string; isUser: boolean}>>([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const chatBottomRef = useRef<HTMLDivElement>(null)
  const activeSearchRequestRef = useRef<number>(0)
  const loadMoreRef = useRef<HTMLDivElement | null>(null)
  const hasSynthesisAnswer = answer.trim() !== ''
  const hasSynthesisWarning = answerWarning.trim() !== ''

  const navigateToPage = (nextPage: AppPage): void => {
    if (nextPage === page) {
      return
    }

    const nextHash = nextPage === 'about' ? '#/about' : '#/'
    window.location.hash = nextHash
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setSelectedResult(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages, chatLoading])

  useEffect(() => {
    const handleHashChange = () => {
      setPage(getCurrentPage())
    }

    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  useEffect(() => {
    if (page === 'about') {
      setSelectedResult(null)
    }
  }, [page])

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

  useEffect(() => {
    const sentinel = loadMoreRef.current
    if (!sentinel || loading || visibleCount >= sortedResults.length) {
      return
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries
        if (entry?.isIntersecting) {
          setVisibleCount((current) => Math.min(current + 12, sortedResults.length))
        }
      },
      {
        root: null,
        rootMargin: '320px 0px',
        threshold: 0.1,
      }
    )

    observer.observe(sentinel)
    return () => observer.disconnect()
  }, [visibleCount, sortedResults.length, loading])

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
    const redditOpt = params.get('reddit') !== 'false'
    const from = params.get('date_from') ?? ''
    const to = params.get('date_to') ?? ''

    setSource(src)
    setSearchMode(mode)
    setTimeFilter(time)
    setAreaFilter(area)
    setIntentFilter(intent)
    setFutureOnly(future)
    setIncludeReddit(redditOpt)
    setDateFrom(from)
    setDateTo(to)

    if (q) setSearchInput(q)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Sync URL whenever search state changes
  useEffect(() => {
    if (page !== 'search') {
      return
    }

    const params = new URLSearchParams()
    if (searchTerm) params.set('q', searchTerm)
    if (source !== 'all') params.set('source', source)
    if (searchMode !== 'svd') params.set('mode', searchMode)
    if (timeFilter !== 'any') params.set('time', timeFilter)
    if (areaFilter !== 'any') params.set('area', areaFilter)
    if (intentFilter !== 'any') params.set('intent', intentFilter)
    if (!futureOnly) params.set('future_only', 'false')
    if (!includeReddit) params.set('reddit', 'false')
    if (dateFrom) params.set('date_from', dateFrom)
    if (dateTo) params.set('date_to', dateTo)
    const qs = params.toString()
    window.history.replaceState(null, '', qs ? `?${qs}` : window.location.pathname)
  }, [page, searchTerm, source, searchMode, timeFilter, areaFilter, intentFilter, futureOnly, includeReddit, dateFrom, dateTo])

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

  const runSearch = async (
    value: string,
    selectedSource: string = source,
    selectedMode: 'svd' | 'tfidf' = searchMode,
    selectedTime: string = timeFilter,
    selectedArea: string = areaFilter,
    selectedIntent: string = intentFilter,
    selectedFutureOnly: boolean = futureOnly,
    selectedIncludeReddit: boolean = includeReddit,
    selectedDateFrom: string = dateFrom,
    selectedDateTo: string = dateTo,
    includeSummary: boolean = false,
    requestId: number = activeSearchRequestRef.current
  ): Promise<boolean> => {
    if (value.trim() === '') {
      activeSearchRequestRef.current += 1
      setSearchTerm('')
      setResults([])
      setVisibleCount(10)
      setAnswer('')
      setAnswerWarning('')
      setQueryLatentProfile({ positive: [], negative: [] })
      setRetrievalContext([])
      setError('')
      setLoading(false)
      setSummaryLoading(false)
      return false
    }

    const { composedQuery, labels } = buildAugmentedQuery(value, selectedTime, selectedArea, selectedIntent)
    if (includeSummary) {
      setSummaryLoading(true)
    } else {
      setVisibleCount(10)
      setSearchTerm(value.trim())
      setError('')
      setRetrievalContext(labels)
      setAnswer('')
      setAnswerWarning('')
      setRewrittenQuery(null)
      setLoading(true)
    }

    try {
      let apiUrl = `/api/search?q=${encodeURIComponent(composedQuery)}&raw_q=${encodeURIComponent(value.trim())}&source=${encodeURIComponent(selectedSource)}&mode=${encodeURIComponent(selectedMode)}&future_only=${selectedFutureOnly}&reddit=${selectedIncludeReddit}&top_k=30`
      if (selectedDateFrom) apiUrl += `&date_from=${encodeURIComponent(selectedDateFrom)}`
      if (selectedDateTo) apiUrl += `&date_to=${encodeURIComponent(selectedDateTo)}`
      apiUrl += `&include_summary=${includeSummary ? '1' : '0'}`
      const response = await fetch(apiUrl)

      if (!response.ok) {
        throw new Error(`Search failed with status ${response.status}`)
      }

      const data = await response.json()
      if (requestId !== activeSearchRequestRef.current) {
        return false
      }

      if (includeSummary) {
        setAnswer(data.answer ?? '')
        setAnswerWarning(data.answer_warning ?? '')
        setRewrittenQuery(data.rewritten_query ?? null)
      } else {
        setResults(data.results ?? [])
        setAnswerWarning('')
        setQueryLatentProfile(data.query_latent_profile ?? { positive: [], negative: [] })
        setEffectiveMode(data.effective_mode ?? selectedMode)
        setRewrittenQuery(data.rewritten_query ?? null)
      }
      return true
    } catch (err) {
      console.error(err)
      if (requestId !== activeSearchRequestRef.current) {
        return false
      }

      if (includeSummary) {
        setAnswer('')
        setAnswerWarning('Failed to load the LLM summary.')
        setRewrittenQuery(null)
      } else {
        setError('Failed to load search results.')
        setResults([])
        setAnswer('')
        setAnswerWarning('')
        setRewrittenQuery(null)
        setQueryLatentProfile({ positive: [], negative: [] })
        setRetrievalContext(labels)
      }
      return false
    } finally {
      if (includeSummary) {
        setSummaryLoading(false)
      } else {
        setLoading(false)
      }
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
    selectedIncludeReddit: boolean = includeReddit,
    selectedDateFrom: string = dateFrom,
    selectedDateTo: string = dateTo
  ): Promise<void> => {
    const requestId = activeSearchRequestRef.current + 1
    activeSearchRequestRef.current = requestId

    const retrievalSucceeded = await runSearch(
      value,
      selectedSource,
      selectedMode,
      selectedTime,
      selectedArea,
      selectedIntent,
      selectedFutureOnly,
      selectedIncludeReddit,
      selectedDateFrom,
      selectedDateTo,
      false,
      requestId
    )

    if (!retrievalSucceeded || requestId !== activeSearchRequestRef.current) {
      return
    }

    void runSearch(
      value,
      selectedSource,
      selectedMode,
      selectedTime,
      selectedArea,
      selectedIntent,
      selectedFutureOnly,
      selectedIncludeReddit,
      selectedDateFrom,
      selectedDateTo,
      true,
      requestId
    )
  }

  const clearFilters = (): void => {
    setSource('all')
    setSearchMode('svd')
    setTimeFilter('any')
    setAreaFilter('any')
    setIntentFilter('any')
    setFutureOnly(true)
    setDateFrom('')
    setDateTo('')
  }

  const sendGeneralChat = async (e: React.FormEvent): Promise<void> => {
    e.preventDefault()
    const text = chatInput.trim()
    if (!text || chatLoading) return

    setChatMessages(prev => [...prev, { text, isUser: true }])
    setChatInput('')
    setChatLoading(true)

    const contextResults = sortedResults.slice(0, Math.min(visibleCount, 15))

    try {
      const response = await fetch('/api/chat/results', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, results: contextResults }),
      })

      if (!response.ok) {
        const data = await response.json()
        setChatMessages(prev => [...prev, { text: 'Error: ' + (data.error || response.status), isUser: false }])
        setChatLoading(false)
        return
      }

      let assistantText = ''
      setChatMessages(prev => [...prev, { text: '', isUser: false }])
      setChatLoading(false)

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.error) {
                setChatMessages(prev => [...prev.slice(0, -1), { text: 'Error: ' + data.error, isUser: false }])
                return
              }
              if (data.content !== undefined) {
                assistantText += data.content
                setChatMessages(prev => [...prev.slice(0, -1), { text: assistantText, isUser: false }])
              }
            } catch { /* ignore malformed lines */ }
          }
        }
      }
    } catch {
      setChatMessages(prev => [...prev, { text: 'Something went wrong. Check the console.', isUser: false }])
      setChatLoading(false)
    }
  }

  const rankingModeLabel = searchMode === 'svd'
    ? effectiveMode === 'svd'
      ? 'Hybrid retrieval with latent semantic reranking'
      : 'Hybrid retrieval with lexical fallback'
    : 'TF-IDF lexical baseline'

  const rankingModeNote = searchMode === 'svd'
    ? effectiveMode === 'svd'
      ? 'This query used lexical retrieval plus latent semantic reranking.'
      : 'This query stayed within the hybrid system, but the ranking fell back to lexical matching because semantic reranking was not reliable for this query.'
    : null

  const searchPageContent = (
    <>
      <div className="input-box">
        <img src={SearchIcon} alt="search" />
        <input
          id="search-input"
          placeholder="Search for things to do in Ithaca..."
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              void handleSearch(searchInput)
            }
          }}
        />
      </div>
      <div className="search-input-hint" aria-live="polite">
        <span className="search-input-hint-text">Type your search, then press</span>
        <kbd className="search-input-kbd">Enter</kbd>
        <span className="search-input-hint-text">to load results and generate the summary</span>
      </div>

      <section className="search-controls-card" aria-label="Search filters">
        <div className="filter-control">
          <label htmlFor="source-filter" className="filter-label">Category</label>
            <select
              id="source-filter"
              className="filter-select"
              value={source}
              onChange={(e) => setSource(e.target.value)}
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
                setTimeFilter(e.target.value)
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
                setAreaFilter(e.target.value)
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
                setIntentFilter(e.target.value)
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
                setFutureOnly(e.target.checked)
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
                setDateFrom(e.target.value)
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
                setDateTo(e.target.value)
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
              }}
            >
              Clear dates
            </button>
          )}
        </div>

        <div className="filter-actions-row">
          <button
            type="button"
            className="clear-filters-button"
            onClick={clearFilters}
          >
            Clear all filters
          </button>
        </div>

        <fieldset className="filter-mode-group">
          <legend className="filter-label">Ranking mode</legend>
          <div className="mode-toggle-row" role="radiogroup" aria-label="Search ranking mode">
            <button
              type="button"
              className={`mode-toggle-button ${searchMode === 'svd' ? 'active' : ''}`}
              aria-pressed={searchMode === 'svd'}
              onClick={() => setSearchMode('svd')}
            >
              <span className="mode-toggle-title">Hybrid SVD Search</span>
              <span className="mode-toggle-subtitle">Latent semantic ranking with TF-IDF fallback</span>
            </button>
            <button
              type="button"
              className={`mode-toggle-button ${searchMode === 'tfidf' ? 'active' : ''}`}
              aria-pressed={searchMode === 'tfidf'}
              onClick={() => setSearchMode('tfidf')}
            >
              <span className="mode-toggle-title">TF-IDF Baseline</span>
              <span className="mode-toggle-subtitle">Exact lexical matching</span>
            </button>
          </div>
        </fieldset>
      </section>
    </>
  )

  const aboutPageContent = (
    <section className="about-page" aria-labelledby="about-title">
      <div className="about-hero-card">
        <p className="about-eyebrow">About SideQuest</p>
        <h2 id="about-title" className="about-title">A hybrid discovery engine for things to do around Cornell and Ithaca</h2>
        <p className="about-lead">
          SideQuest helps students find events, study spots, food, outdoor activities, and local places using a hybrid retrieval system that combines lexical search, SVD-based semantic signals, and an LLM synthesis layer.
        </p>
      </div>

      <div className="about-grid">
        <article className="about-card">
          <h3 className="about-card-title">How to use it</h3>
          <ul className="about-list">
            <li>Type a query like "quiet study spots", "cheap dinner in Collegetown", or "things to do this weekend".</li>
            <li>Press <strong>Enter</strong> to run retrieval and generate the summary card.</li>
            <li>Use the filters to narrow results by category, time, area, vibe, and date range.</li>
            <li>Open a result card to read the full description, hours, reviews, and links when available.</li>
          </ul>
        </article>

        <article className="about-card">
          <h3 className="about-card-title">How ranking works</h3>
          <ul className="about-list">
            <li><strong>Hybrid SVD Search</strong> blends lexical retrieval with latent semantic reranking, which helps surface conceptually related results.</li>
            <li><strong>TF-IDF Baseline</strong> uses exact lexical similarity and is useful as a transparent comparison mode.</li>
            <li>The RAG layer can rewrite vague queries to add useful context, while the system still preserves the original query signal for ranking.</li>
          </ul>
        </article>

        <article className="about-card">
          <h3 className="about-card-title">What the summary means</h3>
          <p className="about-card-copy">
            The LLM synthesis card is grounded in retrieved results. It appears separately from the search results so you can start browsing immediately while the summary is still being generated.
          </p>
          <p className="about-card-copy">
            If no summary appears, retrieval can still work normally and you can inspect the result cards directly.
          </p>
        </article>

        <article className="about-card">
          <h3 className="about-card-title">Good example searches</h3>
          <div className="about-chip-row">
            <span className="about-chip">study spots</span>
            <span className="about-chip">late night food</span>
            <span className="about-chip">weekend activities</span>
            <span className="about-chip">quiet cafes</span>
            <span className="about-chip">outdoor hikes</span>
            <span className="about-chip">cheap eats in Collegetown</span>
          </div>
        </article>
      </div>
    </section>
  )

  return (
    <div className="full-body-container">
      <div className="top-text">
        <div className="logo-container">
          <img src={BearLogo} alt="Cornell Bear Quest Logo" style={{ height: '72px', width: 'auto', mixBlendMode: 'multiply' }} />
          <h1 className="sidequest-title">Side<span>Quest</span></h1>
        </div>
        <nav className="page-nav" aria-label="Primary">
          <button
            type="button"
            className={`page-nav-button ${page === 'search' ? 'active' : ''}`}
            aria-pressed={page === 'search'}
            onClick={() => navigateToPage('search')}
          >
            Search
          </button>
          <button
            type="button"
            className={`page-nav-button ${page === 'about' ? 'active' : ''}`}
            aria-pressed={page === 'about'}
            onClick={() => navigateToPage('about')}
          >
            About / Help
          </button>
        </nav>

        {page === 'search' ? searchPageContent : aboutPageContent}
      </div>

      {page === 'search' && (
      <div id="answer-box">
        {loading && (
          <p className="status-message loading-pulse">
            Searching the area...
          </p>
        )}
        {error && <p className="status-message error-message">{error}</p>}

        {!loading && !error && searchTerm.trim() !== '' && (
          <section className={`synthesis-card ${hasSynthesisWarning && !hasSynthesisAnswer ? 'synthesis-card-warning' : ''}`} aria-live="polite">
            <div className="synthesis-header">
              <div>
                <p className="synthesis-eyebrow">LLM Synthesis</p>
                <h2 className="synthesis-title">Quick recommendation summary</h2>
                <br></br>
              </div>
              <span className={`synthesis-status-pill ${hasSynthesisAnswer ? 'ready' : 'offline'}`}>
                {hasSynthesisAnswer ? 'Available' : summaryLoading ? 'Generating' : 'On demand'}
              </span>
            </div>

            {hasSynthesisAnswer && (
              <div className="synthesis-copy">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
                  }}
                >
                  {answer}
                </ReactMarkdown>
              </div>
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

            {!hasSynthesisAnswer && !hasSynthesisWarning && summaryLoading && (
              <div className="synthesis-empty-state">
                <p className="synthesis-empty-title">Generating summary...</p>
                <p className="synthesis-empty-copy">
                  Retrieved results are ready below while the assistant writes a grounded recommendation.
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
            {rewrittenQuery && (
              <div className="rewritten-query-note" style={{ marginBottom: '12px', padding: '8px', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '6px' }}>
                <strong style={{ color: '#ffb6b6' }}>RAG Query Rewrite:</strong> Using <em>"{rewrittenQuery}"</em>
              </div>
            )}
            <p className="mode-summary-label">
              Search mode in use: <strong>{rankingModeLabel}</strong>
            </p>
            {rankingModeNote && (
              <p className="mode-summary-note">{rankingModeNote}</p>
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
          <section className="results-section">
            <div className="results-toolbar">
              <div className="results-toolbar-copy">
                <p className="results-count">{sortedResults.length} results</p>
              </div>
              <div className="sort-row" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <label className="toggle-switch-label" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', opacity: 0.9 }}>
                  <div className="custom-switch">
                    <input 
                      type="checkbox" 
                      checked={includeReddit}
                      onChange={(e) => {
                        const val = e.target.checked
                        setIncludeReddit(val)
                        activeSearchRequestRef.current += 1
                        if (searchInput.trim()) {
                          runSearch(searchInput, source, searchMode, timeFilter, areaFilter, intentFilter, futureOnly, val, dateFrom, dateTo, false, activeSearchRequestRef.current)
                        }
                      }}
                    />
                    <span className="custom-slider"></span>
                  </div>
                  <span style={{ fontSize: '0.82rem', color: 'var(--cornell-ink-soft)', fontWeight: 700, letterSpacing: '0.02em', textTransform: 'uppercase', userSelect: 'none' }}>Include Reddit</span>
                </label>

                <div style={{ width: '1px', height: '24px', backgroundColor: 'var(--cornell-border)' }}></div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <label htmlFor="sort-select" className="filter-label" style={{ margin: 0 }}>Sort by</label>
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
              </div>
            </div>

            <div className="results-grid">
              {sortedResults.slice(0, visibleCount).map((result) => (
                <div
                  key={result.id}
                  className="episode-item"
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelectedResult(result)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setSelectedResult(result) }}
                >
                  <div className="episode-title-row">
                    <h3 className="episode-title">{result.title}</h3>
                    <div className="episode-title-chips">
                      {result.places_data?.price_level && (
                        <span className="meta-chip price-chip">{result.places_data.price_level}</span>
                      )}
                      {result.places_data?.rating != null && (
                        <span className="meta-chip rating-chip">
                          ★ {result.places_data.rating.toFixed(1)}
                          {result.places_data.rating_count != null && (
                            <span className="rating-count"> ({result.places_data.rating_count.toLocaleString()})</span>
                          )}
                        </span>
                      )}
                    </div>
                  </div>

                  {result.description && (
                    <p className="episode-desc">
                      {result.source === 'reddit' && result.description.length > 150 
                        ? result.description.slice(0, 150) + '...' 
                        : result.description}
                    </p>
                  )}

                  {result.reddit_snippet && (
                    <div style={{ padding: '12px 16px', marginTop: '1rem', marginBottom: '1rem', backgroundColor: '#f9fafb', borderLeft: '3px solid #d1d5db', fontStyle: 'italic', fontSize: '0.95em', color: '#4b5563', borderRadius: '0 8px 8px 0' }}>
                      💡 {result.reddit_snippet.length > 150 ? result.reddit_snippet.slice(0, 150) + '...' : result.reddit_snippet}
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
                      <span className="meta-chip source-chip">{result.source}</span>
                    )}
                    {result.category && (
                      <span className="meta-chip">{result.category}</span>
                    )}
                    {result.location && (
                      <span className="meta-chip">📍 {result.location}</span>
                    )}
                    {result.start_time && (
                      <span className="meta-chip">🕒 {result.start_time}</span>
                    )}
                    {result.organization && (
                      <span className="meta-chip">Host: {result.organization}</span>
                    )}
                  </div>

                  <div className="episode-actions">
                    <span className="meta-chip score-chip">Match: {Math.round(result.score * 100)}%</span>
                    <span className="action-button">View Details</span>
                  </div>
                </div>
              ))}
            </div>

            {!loading && visibleCount < sortedResults.length && (
              <div ref={loadMoreRef} className="results-load-sentinel" aria-hidden="true">
                <span className="results-load-copy">Loading more results as you scroll...</span>
              </div>
            )}
          </section>
        )}
      </div>
      )}

      {page === 'search' && selectedResult && (
        <div className="modal-backdrop" onClick={() => setSelectedResult(null)}>
          <div className="modal-panel" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
            <button className="modal-close" onClick={() => setSelectedResult(null)} aria-label="Close">✕</button>

            {selectedResult.places_data?.photo_url && (
              <img
                className="modal-photo"
                src={selectedResult.places_data.photo_url}
                alt={selectedResult.title}
              />
            )}

            <div className="modal-body">
              <div className="modal-header-row">
                <h2 className="modal-title">{selectedResult.title}</h2>
                <div className="modal-header-chips">
                  {selectedResult.places_data?.price_level && (
                    <span className="meta-chip price-chip">{selectedResult.places_data.price_level}</span>
                  )}
                  {selectedResult.places_data?.rating != null && (
                    <span className="meta-chip rating-chip">
                      ★ {selectedResult.places_data.rating.toFixed(1)}
                      {selectedResult.places_data.rating_count != null && (
                        <span className="rating-count"> ({selectedResult.places_data.rating_count.toLocaleString()})</span>
                      )}
                    </span>
                  )}
                </div>
              </div>

              <p className="modal-meta">
                {[selectedResult.category, selectedResult.location].filter(Boolean).join(' · ')}
              </p>

              <div className="modal-action-row">
                {selectedResult.places_data?.website && (
                  <a href={selectedResult.places_data.website} target="_blank" rel="noreferrer" className="action-button">
                    Visit Website
                  </a>
                )}
                {!selectedResult.places_data?.website && selectedResult.url && (
                  <a href={selectedResult.url} target="_blank" rel="noreferrer" className="action-button">
                    View Details
                  </a>
                )}
                {selectedResult.lat != null && selectedResult.lon != null && (
                  <a
                    href={`https://www.google.com/maps/search/?api=1&query=${selectedResult.lat},${selectedResult.lon}`}
                    target="_blank"
                    rel="noreferrer"
                    className="action-button action-button-outline"
                  >
                    Open in Maps
                  </a>
                )}
              </div>

              {selectedResult.places_data?.phone && (
                <p className="modal-phone">📞 {selectedResult.places_data.phone}</p>
              )}

              {selectedResult.description && (
                <p className="modal-description">{selectedResult.description}</p>
              )}

              {selectedResult.reddit_snippet && (
                <div style={{ padding: '12px 16px', marginTop: '1rem', marginBottom: '1rem', backgroundColor: '#f9fafb', borderLeft: '3px solid #d1d5db', fontStyle: 'italic', fontSize: '0.95em', color: '#4b5563', borderRadius: '0 8px 8px 0' }}>
                  💡 {selectedResult.reddit_snippet}
                </div>
              )}

              {selectedResult.places_data?.hours && (
                <div className="modal-section">
                  <h3 className="modal-section-title">Hours</h3>
                  <ul className="modal-hours-list">
                    {selectedResult.places_data.hours.map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                </div>
              )}

              {selectedResult.places_data?.reviews && selectedResult.places_data.reviews.length > 0 && (
                <div className="modal-section">
                  <h3 className="modal-section-title">Reviews</h3>
                  <div className="modal-reviews">
                    {selectedResult.places_data.reviews.map((review, i) => (
                      <div key={i} className="modal-review">
                        <div className="modal-review-header">
                          <span className="modal-review-author">{review.author}</span>
                          {review.rating != null && (
                            <span className="modal-review-stars">{'★'.repeat(review.rating)}{'☆'.repeat(5 - review.rating)}</span>
                          )}
                          <span className="modal-review-time">{review.relative_time}</span>
                        </div>
                        {review.text && <p className="modal-review-text">{review.text}</p>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

            </div>
          </div>
        </div>
      )}
      {page === 'search' && results.length > 0 && (
        <>
          <button
            className={`chat-fab ${chatOpen ? 'chat-fab-open' : ''}`}
            onClick={() => setChatOpen(o => !o)}
            aria-label={chatOpen ? 'Close chat' : 'Ask about these results'}
          >
            {chatOpen ? (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            )}
          </button>

          {chatOpen && (
            <div className="chat-panel">
              <div className="chat-panel-header">
                <span className="chat-panel-title">Ask about these results</span>
                <button className="chat-panel-close" onClick={() => setChatOpen(false)} aria-label="Close chat">✕</button>
              </div>

              <div className="chat-panel-messages">
                {chatMessages.length === 0 && (
                  <p className="chat-panel-empty">Ask anything about the {sortedResults.length} results on screen — comparisons, recommendations, hours, vibe…</p>
                )}
                {chatMessages.map((msg, i) => (
                  <div key={i} className={`chat-bubble ${msg.isUser ? 'user' : 'assistant'}`}>
                    <p>{msg.text}</p>
                  </div>
                ))}
                {chatLoading && (
                  <div className="loading-indicator visible">
                    <span className="loading-dot" />
                    <span className="loading-dot" />
                    <span className="loading-dot" />
                  </div>
                )}
                <div ref={chatBottomRef} />
              </div>

              <form className="chat-panel-form" onSubmit={sendGeneralChat}>
                <input
                  type="text"
                  placeholder="Ask a question…"
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  disabled={chatLoading}
                  autoComplete="off"
                  autoFocus
                />
                <button type="submit" disabled={chatLoading || !chatInput.trim()}>Send</button>
              </form>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default App
