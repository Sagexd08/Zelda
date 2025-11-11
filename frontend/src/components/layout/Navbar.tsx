import { useLocation } from 'react-router-dom'
import PillNav from './PillNav'
import type { PillNavItem } from './PillNav'

const Navbar = () => {
  const location = useLocation()

  // Use the logo from public folder
  const logo = '/logo.svg'

  const navItems: PillNavItem[] = [
    { label: 'Home', href: '/', ariaLabel: 'Go to home page' },
    { label: 'Register', href: '/register', ariaLabel: 'Register a new user' },
    { label: 'Authenticate', href: '/authenticate', ariaLabel: 'Authenticate user' },
    { label: 'Dashboard', href: '/dashboard', ariaLabel: 'View dashboard' },
    { label: 'Users', href: '/users', ariaLabel: 'View users' },
    { label: 'Logs', href: '/logs', ariaLabel: 'View logs' },
    { label: 'Settings', href: '/settings', ariaLabel: 'View settings' },
  ]

  return (
    <PillNav
      logo={logo}
      logoAlt="Zelda AI Logo"
      items={navItems}
      activeHref={location.pathname}
      className="pill-nav-custom"
      ease="power2.easeOut"
      baseColor="#1E90FF"
      pillColor="rgba(10, 15, 28, 0.85)"
      hoveredPillTextColor="#1E90FF"
      pillTextColor="#ffffff"
      initialLoadAnimation={true}
    />
  )
}

export default Navbar
