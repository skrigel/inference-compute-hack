import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { db } from "../lib/storage";

export function NavBar() {
  const location = useLocation();
  const [lastCorpusId, setLastCorpusId] = useState<string | null>(null);

  useEffect(() => {
    db.preferences.get<string>("lastCorpusId").then((id) => {
      if (id) setLastCorpusId(id);
    });
  }, []);

  const isActive = (path: string) => {
    if (path === "/") return location.pathname === "/";
    return location.pathname.startsWith(path);
  };

  const searchPath = lastCorpusId ? `/search/${lastCorpusId}` : null;
  const onSearchPage = location.pathname.startsWith("/search/");

  return (
    <header className="nav-bar">
      <Link to="/" className="nav-brand">
        grep<span>meaning</span>
      </Link>
      <nav className="nav-tabs">
        <Link to="/" className={`nav-tab${isActive("/") && !onSearchPage && location.pathname !== "/library" ? " active" : ""}`}>
          Home
        </Link>
        <Link to="/library" className={`nav-tab${isActive("/library") ? " active" : ""}`}>
          Library
        </Link>
        {searchPath ? (
          <Link to={searchPath} className={`nav-tab${onSearchPage ? " active" : ""}`}>
            Search
          </Link>
        ) : (
          <span className="nav-tab disabled">Search</span>
        )}
      </nav>
      <div className="nav-spacer" />
      <button className="nav-icon-btn" aria-label="Help">
        <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM8.94 6.94a.75.75 0 11-1.061-1.061 3 3 0 112.871 5.026v.345a.75.75 0 01-1.5 0v-.5c0-.72.57-1.172 1.081-1.287A1.5 1.5 0 108.94 6.94zM10 15a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
        </svg>
      </button>
      <button className="nav-icon-btn" aria-label="Settings">
        <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
          <path fillRule="evenodd" d="M7.84 1.804A1 1 0 018.82 1h2.36a1 1 0 01.98.804l.331 1.652a6.993 6.993 0 011.929 1.115l1.598-.54a1 1 0 011.186.447l1.18 2.044a1 1 0 01-.205 1.251l-1.267 1.113a7.047 7.047 0 010 2.228l1.267 1.113a1 1 0 01.206 1.25l-1.18 2.045a1 1 0 01-1.187.447l-1.598-.54a6.993 6.993 0 01-1.929 1.115l-.33 1.652a1 1 0 01-.98.804H8.82a1 1 0 01-.98-.804l-.331-1.652a6.993 6.993 0 01-1.929-1.115l-1.598.54a1 1 0 01-1.186-.447l-1.18-2.044a1 1 0 01.205-1.251l1.267-1.114a7.05 7.05 0 010-2.227L1.821 7.773a1 1 0 01-.206-1.25l1.18-2.045a1 1 0 011.187-.447l1.598.54A6.993 6.993 0 017.51 3.456l.33-1.652zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
        </svg>
      </button>
    </header>
  );
}
