import { createRoot } from 'react-dom/client';
import 'molstar/build/viewer/molstar.css';
import './styles.css';
import { App } from './App';

createRoot(document.getElementById('root')!).render(<App />);
