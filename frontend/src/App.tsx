import { BrowserRouter, Route, Routes } from "react-router-dom";
import { ToastProvider } from "./components/ui";
import { TourProvider } from "./components/Tour";
import UpdateBanner from "./components/UpdateBanner";
import ProjectsPage from "./pages/ProjectsPage";
import ProjectPage from "./pages/ProjectPage";
import AnnotatorPage from "./pages/AnnotatorPage";

export default function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <TourProvider>
        <UpdateBanner />
        <Routes>
          <Route path="/" element={<ProjectsPage />} />
          <Route path="/projects/:projectId/annotate/:imageId" element={<AnnotatorPage />} />
          <Route path="/projects/:projectId/:tab?" element={<ProjectPage />} />
        </Routes>
        </TourProvider>
      </BrowserRouter>
    </ToastProvider>
  );
}
