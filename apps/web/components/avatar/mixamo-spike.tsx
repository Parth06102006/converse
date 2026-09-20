"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// TODO(spike): replace with a real Mixamo export. Download X Bot or Y Bot
// from https://www.mixamo.com (free Adobe account) as FBX, convert to .glb
// (Blender: import FBX, export glTF with skinning; or FBX2glTF), and place it at
// public/avatars/mixamo-xbot.glb. This placeholder URL is intentional: there is
// no checked-in binary asset and Mixamo downloads require login, so the loader
// path is exercised but the model 404s until the asset is dropped in.
export const DEFAULT_MIXAMO_MODEL_URL = "/avatars/mixamo-xbot.glb";

// Arm pose shaped like the ArmPose interface in webgl-avatar.tsx so recorded
// MediaPipe landmark curves can later drive either rig without reshaping data.
export interface MixamoArmPose {
  shoulderRotation: [number, number, number];
  elbowRotation: [number, number, number];
  wristRotation: [number, number, number];
  fingerCurl: [number, number, number, number, number];
}

export interface MixamoSpikeProps {
  modelUrl?: string;
  leftArm?: MixamoArmPose;
  rightArm?: MixamoArmPose;
  className?: string;
}

const REST_ARM_LEFT: MixamoArmPose = {
  shoulderRotation: [0.35, 0, 0.22],
  elbowRotation: [0.45, 0, 0],
  wristRotation: [0.1, 0, 0],
  fingerCurl: [0.25, 0.25, 0.25, 0.25, 0.25],
};

const REST_ARM_RIGHT: MixamoArmPose = {
  shoulderRotation: [0.35, 0, -0.22],
  elbowRotation: [0.45, 0, 0],
  wristRotation: [0.1, 0, 0],
  fingerCurl: [0.25, 0.25, 0.25, 0.25, 0.25],
};

// Mixamo rig bone names (mixamorig prefix). Hands are single "mitten" bones:
// mixamorigLeftHand / mixamorigRightHand have no finger children, so
// fingerCurl has no target on a stock Mixamo rig (see hand-limitation note).
export const MIXAMO_BONE_MAP = {
  leftUpperArm: "mixamorigLeftArm",
  leftForeArm: "mixamorigLeftForeArm",
  leftHand: "mixamorigLeftHand",
  rightUpperArm: "mixamorigRightArm",
  rightForeArm: "mixamorigRightForeArm",
  rightHand: "mixamorigRightHand",
} as const;

interface MixamoArmBones {
  upperArm: THREE.Bone | null;
  foreArm: THREE.Bone | null;
  hand: THREE.Bone | null;
}

type LoadStatus = "loading" | "ready" | "error";

export function MixamoSpike({
  modelUrl = DEFAULT_MIXAMO_MODEL_URL,
  leftArm = REST_ARM_LEFT,
  rightArm = REST_ARM_RIGHT,
  className,
}: MixamoSpikeProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const bonesRef = useRef<{ left: MixamoArmBones; right: MixamoArmBones }>({
    left: { upperArm: null, foreArm: null, hand: null },
    right: { upperArm: null, foreArm: null, hand: null },
  });
  // Poses are consumed by the render loop; assigning the ref during render
  // keeps the loop at 60fps without re-running effects per pose change.
  const poseRef = useRef<{ left: MixamoArmPose; right: MixamoArmPose }>({
    left: leftArm,
    right: rightArm,
  });
  // Poses are consumed by the render loop; syncing via effect keeps the loop
  // at 60fps without re-running the scene setup per pose change.
  useEffect(() => {
    poseRef.current = { left: leftArm, right: rightArm };
  }, [leftArm, rightArm]);
  const [status, setStatus] = useState<LoadStatus>("loading");

  useEffect(() => {
    const container = mountRef.current;
    if (!container) {
      return;
    }

    const width = container.clientWidth || 400;
    const height = container.clientHeight || 380;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(36, width / height, 0.1, 100);
    camera.position.set(0, 1.42, 2.35);
    camera.lookAt(0, 1.35, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.25);
    keyLight.position.set(1.5, 3.0, 2.5);
    scene.add(keyLight);

    const applyArmPose = (bones: MixamoArmBones, pose: MixamoArmPose) => {
      if (bones.upperArm) {
        bones.upperArm.rotation.set(
          pose.shoulderRotation[0],
          pose.shoulderRotation[1],
          pose.shoulderRotation[2],
        );
      }
      if (bones.foreArm) {
        bones.foreArm.rotation.set(
          pose.elbowRotation[0],
          pose.elbowRotation[1],
          pose.elbowRotation[2],
        );
      }
      if (bones.hand) {
        bones.hand.rotation.set(
          pose.wristRotation[0],
          pose.wristRotation[1],
          pose.wristRotation[2],
        );
      }
      // fingerCurl intentionally has no target: stock Mixamo hands are single
      // mitten bones with no finger joints, so fingerspelling cannot be
      // expressed on this rig without grafted hand bones.
    };

    const findBone = (root: THREE.Object3D, name: string): THREE.Bone | null => {
      const node = root.getObjectByName(name);
      return node instanceof THREE.Bone ? node : null;
    };

    const loader = new GLTFLoader();
    let cancelled = false;
    let animationFrameId = 0;

    loader.load(
      modelUrl,
      (gltf) => {
        if (cancelled) {
          return;
        }
        const model = gltf.scene;

        // Mixamo exports are centimetre-scale Y-up; normalise to ~1.7m tall.
        const bbox = new THREE.Box3().setFromObject(model);
        const size = bbox.getSize(new THREE.Vector3());
        if (size.y > 0) {
          const scale = 1.7 / size.y;
          model.scale.setScalar(scale);
        }
        const centered = new THREE.Box3().setFromObject(model);
        const center = centered.getCenter(new THREE.Vector3());
        model.position.x -= center.x;
        model.position.z -= center.z;
        model.position.y -= centered.min.y;
        scene.add(model);

        // Bind pose of X Bot / Y Bot is a T-pose; resolve arm bones once the
        // skinned mesh (and its skeleton) exists in the scene graph.
        bonesRef.current = {
          left: {
            upperArm: findBone(model, MIXAMO_BONE_MAP.leftUpperArm),
            foreArm: findBone(model, MIXAMO_BONE_MAP.leftForeArm),
            hand: findBone(model, MIXAMO_BONE_MAP.leftHand),
          },
          right: {
            upperArm: findBone(model, MIXAMO_BONE_MAP.rightUpperArm),
            foreArm: findBone(model, MIXAMO_BONE_MAP.rightForeArm),
            hand: findBone(model, MIXAMO_BONE_MAP.rightHand),
          },
        };
        setStatus("ready");
      },
      undefined,
      () => {
        if (!cancelled) {
          setStatus("error");
        }
      },
    );

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      applyArmPose(bonesRef.current.left, poseRef.current.left);
      applyArmPose(bonesRef.current.right, poseRef.current.right);
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelled = true;
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
      bonesRef.current = {
        left: { upperArm: null, foreArm: null, hand: null },
        right: { upperArm: null, foreArm: null, hand: null },
      };
    };
  }, [modelUrl]);

  return (
    <div className={className} role="region" aria-label="Mixamo rig spike">
      <div ref={mountRef} style={{ width: "100%", height: 380 }} />
      <p aria-live="polite">
        {status === "loading" ? `Loading Mixamo model from ${modelUrl}…` : null}
        {status === "ready" ? "Mixamo T-pose ready. Arm posing hooks active." : null}
        {status === "error" ? `No model at ${modelUrl}. Drop a Mixamo .glb there (see TODO).` : null}
      </p>
    </div>
  );
}
