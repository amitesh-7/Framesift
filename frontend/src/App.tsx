import { Routes, Route } from "react-router-dom";
import { HomePage } from "./pages/Home";
import { SearchPage } from "./pages/Search";

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/search" element={<SearchPage />} />
    </Routes>
  );
}

export default App;
