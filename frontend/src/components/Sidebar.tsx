import React from 'react';

interface SidebarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
}

const menuItems = [
  { id: 'overview',   label: 'Overview',              icon: '⊞' },
  { id: 'sources',    label: 'Data Sources',           icon: '◧' },
  { id: 'issues',     label: 'Issue Explorer',         icon: '◎' },
  { id: 'emerging',   label: 'Emerging Issues',        icon: '△',  badge: 'Flagship' },
  { id: 'rootcause',  label: 'Root-Cause',             icon: '⊕',  badge: 'Research' },
  { id: 'analyzer',   label: 'Text Analyzer',          icon: '⌖' },
  { id: 'analytics',  label: 'Analytics',              icon: '╌' },
  { id: 'reports',    label: 'Reports',                icon: '≡' },
  { id: 'research',   label: 'Evaluation & RQ',        icon: '⍻' },
];

export const Sidebar: React.FC<SidebarProps> = ({ currentTab, setCurrentTab }) => {
  return (
    <aside style={{
      width: '228px',
      flexShrink: 0,
      background: 'var(--paper)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      position: 'sticky',
      top: 0,
      overflowY: 'auto',
    }}>
      {/* Brand */}
      <div style={{
        padding: '20px 20px 16px',
        borderBottom: '1px solid var(--border)',
      }}>
        {/* Logo mark — simple hand-drawn circle + wave feel */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '6px',
            background: 'var(--teal)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff',
            fontSize: '15px',
            fontWeight: 700,
            letterSpacing: '-1px',
            flexShrink: 0,
          }}>
            VM
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '15px', color: 'var(--text-main)', lineHeight: 1.1 }}>
              VoiceMiner
            </div>
            <div style={{ fontSize: '10px', color: 'var(--text-light)', marginTop: '2px', fontWeight: 500 }}>
              v1.0 · Research Platform
            </div>
          </div>
        </div>
      </div>

      {/* Navigation label */}
      <div style={{
        padding: '14px 20px 6px',
        fontSize: '10px',
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        color: 'var(--text-light)',
      }}>
        Navigation
      </div>

      {/* Menu items */}
      <nav style={{ flex: 1, padding: '0 10px' }}>
        {menuItems.map((item) => {
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setCurrentTab(item.id)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 10px',
                marginBottom: '2px',
                borderRadius: '5px',
                border: 'none',
                cursor: 'pointer',
                fontSize: '13.5px',
                fontWeight: isActive ? 600 : 400,
                color: isActive ? 'var(--teal)' : 'var(--text-muted)',
                background: isActive ? 'var(--teal-light)' : 'transparent',
                textAlign: 'left',
                transition: 'background 0.12s, color 0.12s',
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  (e.currentTarget as HTMLButtonElement).style.background = '#ede8e1';
                  (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-main)';
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  (e.currentTarget as HTMLButtonElement).style.background = 'transparent';
                  (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-muted)';
                }
              }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '9px' }}>
                <span style={{
                  fontFamily: 'monospace',
                  fontSize: '14px',
                  width: '18px',
                  display: 'inline-block',
                  textAlign: 'center',
                  color: isActive ? 'var(--teal)' : 'var(--text-light)',
                }}>
                  {item.icon}
                </span>
                {item.label}
              </span>
              {item.badge && (
                <span style={{
                  fontSize: '9px',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  padding: '1px 5px',
                  borderRadius: '3px',
                  background: isActive ? 'var(--teal)' : '#e8e2d8',
                  color: isActive ? '#fff' : 'var(--text-muted)',
                  border: isActive ? 'none' : '1px solid var(--border)',
                }}>
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer status */}
      <div style={{
        padding: '14px 20px',
        borderTop: '1px solid var(--border)',
        fontSize: '11px',
        color: 'var(--text-light)',
      }}>
        <div style={{ fontWeight: 600, color: 'var(--green)', marginBottom: '4px', fontSize: '11px' }}>
          ● Pipeline Operational
        </div>
        <div style={{ lineHeight: 1.6, fontSize: '10.5px' }}>
          Sentence-BERT · CardiffNLP RoBERTa<br />
          NetworkX Association Graph
        </div>
      </div>
    </aside>
  );
};
