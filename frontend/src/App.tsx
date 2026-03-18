import { useState, useEffect } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { Episode } from './types'
import Chat from './Chat'

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [episodes, setEpisodes] = useState<Episode[]>([])

  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(data => setUseLlm(data.use_llm))
  }, [])

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)
    if (value.trim() === '') { setEpisodes([]); return }
    const response = await fetch(`/api/episodes?title=${encodeURIComponent(value)}`)
    const data: Episode[] = await response.json()
    setEpisodes(data)
  }

  if (useLlm === null) return <></>

  return (
    <div className={`full-body-container ${useLlm ? 'llm-mode' : ''}`}>
      {/* Search bar (always shown) */}
      <div className="top-text">
        <h1 className="sidequest-title">SideQuest</h1>
        <div className="input-box" onClick={() => document.getElementById('search-input')?.focus()}>
          <img src={SearchIcon} alt="search" />
          <input
            id="search-input"
            placeholder="Search for things to do in Ithaca..."
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Search results (always shown) */}
      <div id="answer-box">
        {episodes.map((episode, index) => (
          <div key={index} className="episode-item">
            <h3 className="episode-title">{episode.title}</h3>
            <p className="episode-desc">{episode.descr}</p>
            <p className="episode-rating">IMDB Rating: {episode.imdb_rating}</p>
          </div>
        ))}
      </div>

      {/* Chat (only when USE_LLM = True in routes.py) */}
      {useLlm && <Chat onSearchTerm={handleSearch} />}
    </div>
  )
}

export default App
