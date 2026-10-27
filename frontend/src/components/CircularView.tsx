import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export interface CircularParams {
  nondecision: number; threshold: number; drift: number; direction: number;
  noise: number; seed: number; max_time: number;
}
export interface CircularResult {
  path: [number, number, number][]; hit: boolean; decision_time: number | null;
  response_time: number | null; angle: number | null; dt: number; params: CircularParams;
}

export function CircularView({ result, index }: { result: CircularResult; index: number }) {
  const mount = useRef<HTMLDivElement>(null);
  const state = useRef<{ geometry: THREE.BufferGeometry; marker: THREE.Mesh } | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const host = mount.current!;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true }); }
    catch { setError("3D rendering is unavailable in this browser. Simulation results and playback times are still available."); return; }
    setError("");
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    const radius = result.params.threshold;
    const end = result.path[result.path.length - 1][1];
    // Preserve numerical seconds on Y; stretch only its display scale for readability.
    const timeScale = 3 * radius / Math.max(end, 0.001);
    const height = end * timeScale;
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
    camera.position.set(radius * 4, height * 0.85, radius * 5);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, height / 2, 0);
    controls.enableDamping = true;
    const cylinder = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, height, 96, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
    cylinder.position.y = height / 2;
    scene.add(cylinder);
    function line(points: THREE.Vector3[], color: number) {
      const obj = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color }));
      scene.add(obj); return obj;
    }
    function label(text: string, x: number, y: number, z: number) {
      const canvas = document.createElement("canvas"); canvas.width = 512; canvas.height = 64;
      const ctx = canvas.getContext("2d")!;
      ctx.font = "28px sans-serif"; ctx.fillStyle = "#e2e8f0"; ctx.fillText(text, 4, 42);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
      sprite.position.set(x, y, z); sprite.scale.set(radius * 1.8, radius * 0.225, 1); scene.add(sprite);
    }
    const levels = [0, result.params.nondecision, end];
    levels.forEach((t, i) => {
      const y = t * timeScale;
      line(Array.from({ length: 97 }, (_, j) => new THREE.Vector3(radius * Math.cos(j * Math.PI / 48), y, radius * Math.sin(j * Math.PI / 48))), i === 1 ? 0xfbbf24 : 0x475569);
      label(`${t.toFixed(3)} s${i === 1 ? " · start" : ""}`, -radius * 1.8, y, 0);
    });
    for (let j = 0; j < 8; j++) {
      const x = radius * Math.cos(j * Math.PI / 4), z = radius * Math.sin(j * Math.PI / 4);
      line([new THREE.Vector3(x, 0, z), new THREE.Vector3(x, height, z)], 0x334155);
    }
    line([new THREE.Vector3(-radius, 0, 0), new THREE.Vector3(radius * 1.3, 0, 0)], 0xf87171);
    line([new THREE.Vector3(0, 0, -radius), new THREE.Vector3(0, 0, radius * 1.3)], 0x4ade80);
    line([new THREE.Vector3(-radius * 1.2, 0, 0), new THREE.Vector3(-radius * 1.2, height * 1.15, 0)], 0x94a3b8);
    label("+X · evidence", radius * 1.6, 0, 0);
    label("+Z · evidence", 0, 0, radius * 1.6);
    label("Y · time (s)", -radius * 1.2, height * 1.2, 0);
    line([new THREE.Vector3(), new THREE.Vector3(0, result.params.nondecision * timeScale, 0)], 0x64748b);
    const theta = result.params.direction * Math.PI / 180;
    if (result.params.drift > 0) scene.add(new THREE.ArrowHelper(new THREE.Vector3(Math.cos(theta), 0, Math.sin(theta)),
      new THREE.Vector3(0, result.params.nondecision * timeScale, 0), radius * 0.65, 0x38bdf8));
    const trajectory = line(result.path.map(([x, t, z]) => new THREE.Vector3(x, t * timeScale, z)), 0xfbbf24);
    trajectory.geometry.setDrawRange(0, 1);
    const marker = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.035, 16, 16), new THREE.MeshBasicMaterial({ color: 0xfbbf24 }));
    scene.add(marker);
    state.current = { geometry: trajectory.geometry, marker };
    marker.position.set(0, result.params.nondecision * timeScale, 0);
    const resize = () => { const w = host.clientWidth || 800; renderer.setSize(w, 520); camera.aspect = w / 520; camera.updateProjectionMatrix(); };
    const observer = new ResizeObserver(resize); observer.observe(host); resize();
    let raf = 0;
    const animate = () => { controls.update(); renderer.render(scene, camera); raf = requestAnimationFrame(animate); }; animate();
    return () => {
      cancelAnimationFrame(raf); observer.disconnect(); controls.dispose();
      scene.traverse((o) => {
        const obj = o as THREE.Mesh;
        obj.geometry?.dispose();
        const materials = obj.material ? (Array.isArray(obj.material) ? obj.material : [obj.material]) : [];
        materials.forEach((m) => { (m as THREE.SpriteMaterial).map?.dispose(); m.dispose(); });
      });
      renderer.dispose(); renderer.domElement.remove(); state.current = null;
    };
  }, [result]);
  useEffect(() => {
    const s = state.current; if (!s) return;
    s.geometry.setDrawRange(0, index + 1);
    const positions = s.geometry.getAttribute("position");
    s.marker.position.fromBufferAttribute(positions, index);
  }, [index, result]);
  return <><div ref={mount} aria-label="Circular diffusion trajectory: X-Z evidence plane, vertical Y time" />{error && <p role="alert" className="p-3 text-red-700">{error}</p>}</>;
}
