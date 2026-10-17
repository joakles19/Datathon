export default function AlertBanner({ alert, onAck }) {
  if (!alert) return null;
  return (
    <div className="banner" role="alert">
      <span>
        {alert.message || 'Sound alert'} · {alert.label.replace(/_/g, ' ')} (score {Math.round(alert.confidence * 100)}%) ·{' '}
        {new Date(alert.time).toLocaleTimeString()}
      </span>
      <button onClick={onAck}>Acknowledge</button>
    </div>
  );
}