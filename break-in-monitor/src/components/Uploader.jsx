import { useEffect, useRef, useState } from 'react';

const AUDIO_EXT = /\.(wav|mp3|m4a|ogg|webm|flac|aac)$/i;

export default function Uploader({ onFile, onReject }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const [previewName, setPreviewName] = useState('');

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const handle = files => {
    for (const f of files) {
      if (!f.type.startsWith('audio/') && !AUDIO_EXT.test(f.name)) {
        onReject(f.name);
        continue;
      }
      setPreview(f ? URL.createObjectURL(f) : null);
      setPreviewName(f.name);
      onFile(f);
    }
  };

  return (
    <section>
      <h2>Upload audio</h2>
      <div
        className={`drop${over ? ' over' : ''}`}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current.click()}
        onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && inputRef.current.click()}
        onDragEnter={e => { e.preventDefault(); setOver(true); }}
        onDragOver={e => { e.preventDefault(); setOver(true); }}
        onDragLeave={e => { e.preventDefault(); setOver(false); }}
        onDrop={e => { e.preventDefault(); setOver(false); handle(Array.from(e.dataTransfer.files)); }}
      >
        Drop audio files here or click to browse
        <br />
        <span className="meta">wav, mp3, m4a, ogg, webm, flac</span>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        multiple
        hidden
        onChange={e => { handle(Array.from(e.target.files)); e.target.value = ''; }}
      />
      {preview && (
        <>
          <p className="meta" style={{ margin: '10px 0 0' }}>{previewName}</p>
          <audio src={preview} controls />
        </>
      )}
    </section>
  );
}
