/**
 * MathJax 配置 —— 必须在 tex-mml-chtml.js 之前加载。
 *
 * pymdownx.arithmatex (generic: true) 输出的是被包过的 \[...\]（块级）
 * 和 \(...\)（行内），MathJax 默认只认裸 $...$，所以必须在这里显式
 * 告诉它去认 \[ \] 和 \( \)，否则公式会以纯文本形式显示。
 */
window.MathJax = {
  tex: {
    inlineMath: [
      ['\\(', '\\)']
    ],
    displayMath: [
      ['\\[', '\\]']
    ]
  },
  options: {
    // 跳过页面里的导航/侧栏，只渲染正文
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }
};
