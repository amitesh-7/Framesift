import { Routes, Route } from "react-router-dom";
import {
  HomePage,
  SearchPage,
  FeaturesPage,
  HowItWorksPage,
  TechnologyPage,
  AdminPage,
} from "./pages";

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/search" element={<SearchPage />} />
      <Route path="/features" element={<FeaturesPage />} />
      <Route path="/how-it-works" element={<HowItWorksPage />} />
      <Route path="/technology" element={<TechnologyPage />} />
      <Route path="/admin" element={<AdminPage />} />
    </Routes>
  );
}

export default App;
