import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { FaPlay, FaPause, FaSkipForward, FaSkipBackward, FaVolumeUp, FaHeart, FaSearch } from 'react-icons/fa';
import './App.css';

const AppContainer = styled.div`
  display: flex;
  height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-family: 'Arial', sans-serif;
`;

const Sidebar = styled.div`
  width: 250px;
  background: rgba(0, 0, 0, 0.3);
  padding: 2rem;
  border-right: 1px solid rgba(255, 255, 255, 0.1);
`;

const MainContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
`;

const Header = styled.header`
  padding: 2rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`;

const SearchBar = styled.div`
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 25px;
  padding: 0.5rem 1rem;
  margin-top: 1rem;
`;

const SearchInput = styled.input`
  background: none;
  border: none;
  color: white;
  margin-left: 0.5rem;
  flex: 1;
  outline: none;
  &::placeholder { color: rgba(255, 255, 255, 0.7); }
`;

const MusicGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1.5rem;
  padding: 2rem;
  flex: 1;
  overflow-y: auto;
`;

const TrackCard = styled.div`
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1.5rem;
  cursor: pointer;
  transition: transform 0.2s, background 0.2s;
  &:hover {
    transform: translateY(-5px);
    background: rgba(255, 255, 255, 0.2);
  }
`;

const PlayerBar = styled.div`
  background: rgba(0, 0, 0, 0.5);
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const PlayerControls = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const ControlButton = styled.button`
  background: none;
  border: none;
  color: white;
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 50%;
  transition: background 0.2s;
  &:hover { background: rgba(255, 255, 255, 0.1); }
`;

const PlayButton = styled(ControlButton)`
  font-size: 2rem;
  background: #1db954;
  &:hover { background: #1ed760; }
`;

const TrackInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const VolumeControl = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const VolumeSlider = styled.input`
  width: 100px;
`;

function App() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTrack, setCurrentTrack] = useState(null);
  const [volume, setVolume] = useState(50);
  const [searchTerm, setSearchTerm] = useState('');
  const audioRef = useRef(null);

  // Sample music data (in real app, this would come from API)
  const tracks = [
    { id: 1, title: 'Autonomous Beats', artist: 'TARS AI', album: 'Digital Dreams', duration: '3:45' },
    { id: 2, title: 'Synthetic Symphony', artist: 'Neural Network', album: 'Machine Learning', duration: '4:12' },
    { id: 3, title: 'Code Melody', artist: 'Algorithm', album: 'Programming Sounds', duration: '3:28' },
    { id: 4, title: 'Binary Rhythm', artist: 'Data Stream', album: 'Digital Flow', duration: '4:01' },
    { id: 5, title: 'AI Harmony', artist: 'Superintelligence', album: 'Future Music', duration: '3:55' },
    { id: 6, title: 'Quantum Beats', artist: 'Quantum AI', album: 'Parallel Universe', duration: '4:33' }
  ];

  const filteredTracks = tracks.filter(track =>
    track.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    track.artist.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const playTrack = (track) => {
    setCurrentTrack(track);
    setIsPlaying(true);
  };

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  return (
    <AppContainer>
      <Sidebar>
        <h2>🎵 Music Streaming</h2>
        <div style={{ marginTop: '2rem' }}>
          <h3>Playlists</h3>
          <ul style={{ listStyle: 'none', padding: 0, marginTop: '1rem' }}>
            <li style={{ padding: '0.5rem 0', cursor: 'pointer' }}>🎧 My Favorites</li>
            <li style={{ padding: '0.5rem 0', cursor: 'pointer' }}>🔥 Trending</li>
            <li style={{ padding: '0.5rem 0', cursor: 'pointer' }}>🎸 Rock</li>
            <li style={{ padding: '0.5rem 0', cursor: 'pointer' }}>🎹 Electronic</li>
            <li style={{ padding: '0.5rem 0', cursor: 'pointer' }}>🎤 Pop</li>
          </ul>
        </div>
      </Sidebar>

      <MainContent>
        <Header>
          <h1>Discover Music</h1>
          <SearchBar>
            <FaSearch />
            <SearchInput
              placeholder="Search for songs, artists, or albums..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </SearchBar>
        </Header>

        <MusicGrid>
          {filteredTracks.map(track => (
            <TrackCard key={track.id} onClick={() => playTrack(track)}>
              <div style={{ background: 'rgba(255,255,255,0.1)', height: '150px', borderRadius: '8px', marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '2rem' }}>🎵</div>
              <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.1rem' }}>{track.title}</h3>
              <p style={{ margin: '0 0 0.5rem 0', opacity: 0.8 }}>{track.artist}</p>
              <p style={{ margin: 0, opacity: 0.6, fontSize: '0.9rem' }}>{track.album} • {track.duration}</p>
            </TrackCard>
          ))}
        </MusicGrid>

        <PlayerBar>
          <TrackInfo>
            {currentTrack && (
              <>
                <div style={{ width: '50px', height: '50px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>🎵</div>
                <div>
                  <div style={{ fontWeight: 'bold' }}>{currentTrack.title}</div>
                  <div style={{ opacity: 0.8, fontSize: '0.9rem' }}>{currentTrack.artist}</div>
                </div>
              </>
            )}
          </TrackInfo>

          <PlayerControls>
            <ControlButton><FaSkipBackward /></ControlButton>
            <PlayButton onClick={togglePlayPause}>
              {isPlaying ? <FaPause /> : <FaPlay />}
            </PlayButton>
            <ControlButton><FaSkipForward /></ControlButton>
          </PlayerControls>

          <VolumeControl>
            <FaVolumeUp />
            <VolumeSlider
              type="range"
              min="0"
              max="100"
              value={volume}
              onChange={(e) => setVolume(e.target.value)}
            />
          </VolumeControl>
        </PlayerBar>
      </MainContent>
    </AppContainer>
  );
}

export default App;