import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './App.css'
import { RobotProvider } from './context/RobotContext.jsx'
import { ThemeProvider } from './context/ThemeContext.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <ThemeProvider>
    <RobotProvider>
      <App />
    </RobotProvider>
  </ThemeProvider>,
)
