"use client";

import React, { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";

mermaid.initialize({
  startOnLoad: false,
  theme: "default",
  securityLevel: "loose",
});

export default function Mermaid({ chart }) {
  const ref = useRef(null);
  const [svg, setSvg] = useState("");

  useEffect(() => {
    if (chart) {
      mermaid.render(`mermaid-svg-${Math.random().toString(36).substring(2, 9)}`, chart).then((result) => {
        setSvg(result.svg);
      }).catch((e) => {
        console.error("Mermaid parsing error", e);
      });
    }
  }, [chart]);

  return <div className="mermaid-container" ref={ref} dangerouslySetInnerHTML={{ __html: svg }} />;
}
