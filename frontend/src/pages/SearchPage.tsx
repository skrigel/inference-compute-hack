import { useParams } from "react-router-dom";

export function SearchPage() {
  const { corpusId } = useParams<{ corpusId: string }>();
  return (
    <main className="page-content">
      <h1>Search: {corpusId}</h1>
      <p>Search interface coming soon...</p>
    </main>
  );
}
