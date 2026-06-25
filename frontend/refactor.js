const fs = require('fs');
const path = require('path');

const pagePath = path.join(__dirname, 'src', 'app', 'page.tsx');
let content = fs.readFileSync(pagePath, 'utf8');

const replacements = {
  'bg-\\[#000000\\]': 'bg-background',
  'bg-\\[#050505\\]': 'bg-panel',
  'bg-\\[#0a0a0a\\]': 'bg-panel',
  'bg-\\[#111\\]': 'bg-panel',
  'text-zinc-300': 'text-foreground',
  'text-white': 'text-primary',
  'text-zinc-500': 'text-muted-foreground',
  'text-zinc-400': 'text-muted-foreground',
  'text-zinc-600': 'text-muted-foreground',
  'border-zinc-900': 'border-border',
  'border-zinc-800': 'border-panel-border',
  'border-zinc-700': 'border-panel-border',
  'bg-zinc-900': 'bg-muted',
  'bg-zinc-800': 'bg-accent',
  'text-zinc-200': 'text-foreground',
  'bg-zinc-700': 'bg-accent',
  'text-zinc-100': 'text-foreground'
};

for (const [key, value] of Object.entries(replacements)) {
  const regex = new RegExp(key, 'g');
  content = content.replace(regex, value);
}

fs.writeFileSync(pagePath, content);
console.log('Replaced colors successfully!');
