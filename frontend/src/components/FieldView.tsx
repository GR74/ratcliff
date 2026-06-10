import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { FieldResponse } from "../lib/types";

interface Props {
  field: FieldResponse;
  frameIndex: number;
  showThreshold: boolean;
}

/**
 * Renders one frame of the evidence accumulator field as a 3D height surface,
 * with the decision threshold drawn as a translucent plane. The surface mesh
 * is built once per `field` payload; only vertex z-heights + colors are updated
 * when `frameIndex` changes, which is cheap on the GPU.
 */
export function FieldView({ field, frameIndex, showThreshold }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    geometry: THREE.PlaneGeometry;
    mesh: THREE.Mesh;
    thresholdPlane: THREE.Mesh;
    raf: number;
    heightScale: number;
    globalMin: number;
    globalMax: number;
  } | null>(null);

  // (Re)build the scene whenever the field payload changes.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 800;
    const height = 500;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
    camera.position.set(field.m * 0.9, field.m * 0.7, field.n * 1.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.innerHTML = "";
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dir = new THREE.DirectionalLight(0xffffff, 0.6);
    dir.position.set(1, 1, 1);
    scene.add(dir);

    // Surface: PlaneGeometry with (m-1) x (n-1) segments so vertex count = m*n.
    const geometry = new THREE.PlaneGeometry(field.m, field.n, field.m - 1, field.n - 1);
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      flatShading: false,
      metalness: 0.1,
      roughness: 0.85,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2; // lay the plane flat, height becomes world-Y
    scene.add(mesh);

    // Compute global min/max across all frames for a stable height + color scale.
    let globalMin = Infinity;
    let globalMax = -Infinity;
    for (const frame of field.frames) {
      for (const row of frame) {
        for (const v of row) {
          if (v < globalMin) globalMin = v;
          if (v > globalMax) globalMax = v;
        }
      }
    }
    const span = Math.max(globalMax - globalMin, 1e-6);
    const heightScale = (field.m * 0.5) / span; // surface ~half the grid width tall

    // Threshold plane (translucent) at the threshold height.
    const thrGeo = new THREE.PlaneGeometry(field.m, field.n);
    const thrMat = new THREE.MeshBasicMaterial({
      color: 0xf43f5e,
      transparent: true,
      opacity: 0.18,
      side: THREE.DoubleSide,
    });
    const thresholdPlane = new THREE.Mesh(thrGeo, thrMat);
    thresholdPlane.rotation.x = -Math.PI / 2;
    thresholdPlane.position.y = (field.threshold - globalMin) * heightScale;
    scene.add(thresholdPlane);

    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      stateRef.current!.raf = requestAnimationFrame(animate);
    };

    stateRef.current = {
      renderer,
      scene,
      camera,
      controls,
      geometry,
      mesh,
      thresholdPlane,
      raf: 0,
      heightScale,
      globalMin,
      globalMax,
    };
    stateRef.current.raf = requestAnimationFrame(animate);

    const onResize = () => {
      const w = mount.clientWidth || 800;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(stateRef.current!.raf);
      controls.dispose();
      geometry.dispose();
      material.dispose();
      thrGeo.dispose();
      thrMat.dispose();
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
      stateRef.current = null;
    };
  }, [field]);

  // Update vertex heights + colors when the frame changes.
  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    const frame = field.frames[frameIndex];
    if (!frame) return;

    const pos = st.geometry.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(field.n * field.m * 3);
    const color = new THREE.Color();
    const span = Math.max(st.globalMax - st.globalMin, 1e-6);

    let vi = 0;
    // PlaneGeometry vertex order is row-major over (n rows, m cols).
    for (let r = 0; r < field.n; r++) {
      for (let c = 0; c < field.m; c++) {
        const v = frame[r][c];
        const z = (v - st.globalMin) * st.heightScale;
        pos.setZ(vi, z);
        // viridis-ish: map normalized height to hue 0.7→0.0 (blue→yellow→red)
        const t = (v - st.globalMin) / span;
        color.setHSL(0.7 - 0.7 * t, 0.75, 0.35 + 0.25 * t);
        colors[vi * 3] = color.r;
        colors[vi * 3 + 1] = color.g;
        colors[vi * 3 + 2] = color.b;
        vi++;
      }
    }
    pos.needsUpdate = true;
    st.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    st.geometry.computeVertexNormals();
  }, [field, frameIndex]);

  // Toggle the threshold plane visibility without rebuilding the scene.
  useEffect(() => {
    const st = stateRef.current;
    if (st) st.thresholdPlane.visible = showThreshold;
  }, [showThreshold, field]);

  return <div ref={mountRef} className="w-full rounded overflow-hidden" style={{ height: 500 }} />;
}
