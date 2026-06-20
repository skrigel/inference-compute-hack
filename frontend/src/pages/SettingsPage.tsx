import { useSettings } from "../hooks/useSettings";

export function SettingsPage() {
  const { apiMode, setApiMode } = useSettings();

  return (
    <main className="page-content">
      <div className="page-header">
        <h1 className="page-title">Settings</h1>
      </div>

      <section className="settings-section">
        <h2 className="section-title">Developer Options</h2>

        <label className="setting-row">
          <div className="setting-info">
            <span className="setting-label">Use live backend</span>
            <span className="setting-description">
              Connect to real backend at localhost:8000 instead of mock data
            </span>
          </div>
          <input
            type="checkbox"
            className="setting-toggle"
            checked={apiMode === "live"}
            onChange={(e) => setApiMode(e.target.checked ? "live" : "mock")}
          />
        </label>

        <div className="setting-status">
          Current mode: <strong>{apiMode}</strong>
        </div>
      </section>
    </main>
  );
}
