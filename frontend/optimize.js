const fs = require('fs');
const path = require('path');

const pagePath = path.join(__dirname, 'src', 'app', 'page.tsx');
let content = fs.readFileSync(pagePath, 'utf8');

// 1. Replace transition-all with transition-colors or transition
content = content.replace(/transition-all/g, 'transition');

// 2. Remove backdrop-blur-sm from modal to improve FPS
content = content.replace(/bg-black\/60 backdrop-blur-sm/g, 'bg-black/80');

// 3. Optimize Framer Motion on the modal by adding hardware acceleration classes and removing scale if it stutters, 
// or simply ensuring no layout thrashing.
// Scale animations are usually fine, but let's keep them.

fs.writeFileSync(pagePath, content);
console.log('Optimized page.tsx');
