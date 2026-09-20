"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import type {
  SignRepresentation,
  SignRepresentationToken,
} from "@converse/contracts";
import { blendPoses } from "./coarticulation";
import { facePoseForToken } from "./nmm";
import styles from "./avatar.module.css";
import { handshapeForToken } from "./handshapes";

export interface WebglAvatarProps {
  /** The incoming ASL SignRepresentation from the translation engine */
  representation?: SignRepresentation | null;
  /** Playback speed multiplier (default 1.0) */
  playbackSpeed?: number;
  /** Whether the avatar should execute breathing and idle micro-motions */
  enableIdleMotion?: boolean;
  /** Callback when the current representation finishes signing */
  onPlaybackComplete?: () => void;
  /** Optional container class name */
  className?: string;
}

interface ArmPose {
  shoulderRotation: [number, number, number];
  elbowRotation: [number, number, number];
  wristRotation: [number, number, number];
  fingerCurl: [number, number, number, number, number];
}

export interface AvatarPose {
  headRotation: [number, number, number];
  eyebrowRaise: number;
  eyebrowFurrow: number;
  mouthOpen: number;
  mouthSmile: number;
  mouthWidth: number;
  chestPitch: number;
  leftArm: ArmPose;
  rightArm: ArmPose;
}

const REST_ARM_LEFT: ArmPose = {
  shoulderRotation: [0.35, 0, 0.22],
  elbowRotation: [0.45, 0, 0],
  wristRotation: [0.1, 0, 0],
  fingerCurl: [0.25, 0.25, 0.25, 0.25, 0.25],
};

const REST_ARM_RIGHT: ArmPose = {
  shoulderRotation: [0.35, 0, -0.22],
  elbowRotation: [0.45, 0, 0],
  wristRotation: [0.1, 0, 0],
  fingerCurl: [0.25, 0.25, 0.25, 0.25, 0.25],
};

const REST_POSE: AvatarPose = {
  headRotation: [0, 0, 0],
  eyebrowRaise: 0,
  eyebrowFurrow: 0,
  mouthOpen: 0,
  mouthSmile: 0.1,
  mouthWidth: 1,
  chestPitch: 0,
  leftArm: REST_ARM_LEFT,
  rightArm: REST_ARM_RIGHT,
};

const GLOSS_POSES: Record<string, Partial<AvatarPose>> = {
  HELLO: {
    headRotation: [0.06, 0.05, 0],
    eyebrowRaise: 0.8,
    mouthSmile: 0.5,
    rightArm: {
      shoulderRotation: [-1.25, 0.35, 0.5],
      elbowRotation: [1.85, 0.2, -0.35],
      wristRotation: [-0.2, 0.3, 0.1],
      fingerCurl: [0, 0, 0, 0, 0],
    },
  },
  WELCOME: {
    headRotation: [0.08, 0, 0],
    eyebrowRaise: 0.7,
    mouthSmile: 0.6,
    leftArm: {
      shoulderRotation: [-0.65, -0.35, -0.25],
      elbowRotation: [1.1, -0.45, 0.5],
      wristRotation: [-0.2, 0.2, 0],
      fingerCurl: [0, 0, 0, 0, 0],
    },
    rightArm: {
      shoulderRotation: [-0.65, 0.35, 0.25],
      elbowRotation: [1.1, 0.45, -0.5],
      wristRotation: [-0.2, -0.2, 0],
      fingerCurl: [0, 0, 0, 0, 0],
    },
  },
  THANK_YOU: {
    headRotation: [0.1, 0, 0],
    eyebrowRaise: 0.5,
    mouthOpen: 0.2,
    mouthSmile: 0.4,
    rightArm: {
      shoulderRotation: [-1.15, 0.1, 0.28],
      elbowRotation: [2.15, 0.2, -0.25],
      wristRotation: [-0.2, 0.1, 0],
      fingerCurl: [0, 0, 0, 0, 0],
    },
  },
  THANKS: {
    headRotation: [0.1, 0, 0],
    eyebrowRaise: 0.5,
    mouthSmile: 0.4,
    rightArm: {
      shoulderRotation: [-1.15, 0.1, 0.28],
      elbowRotation: [2.15, 0.2, -0.25],
      wristRotation: [-0.2, 0.1, 0],
      fingerCurl: [0, 0, 0, 0, 0],
    },
  },
  PLEASE: {
    headRotation: [0.05, 0, 0],
    eyebrowRaise: 0.4,
    rightArm: {
      shoulderRotation: [-0.85, -0.2, 0.35],
      elbowRotation: [1.95, -0.35, -0.15],
      wristRotation: [-0.1, 0, 0],
      fingerCurl: [0, 0, 0, 0, 0],
    },
  },
  YES: {
    headRotation: [0.18, 0, 0],
    eyebrowRaise: 0.4,
    rightArm: {
      shoulderRotation: [-0.75, 0.2, 0.25],
      elbowRotation: [1.55, 0, 0],
      wristRotation: [0.55, 0, 0],
      fingerCurl: [1, 1, 1, 1, 1],
    },
  },
  NO: {
    headRotation: [0, 0.25, 0],
    eyebrowFurrow: 0.85,
    rightArm: {
      shoulderRotation: [-0.82, 0.2, 0.28],
      elbowRotation: [1.6, 0, -0.18],
      wristRotation: [0.1, 0, 0],
      fingerCurl: [0.9, 0.85, 0.85, 1, 1],
    },
  },
  WHAT: {
    headRotation: [0.08, 0, 0.05],
    eyebrowFurrow: 0.9,
    mouthOpen: 0.3,
    leftArm: {
      shoulderRotation: [-0.55, -0.25, -0.3],
      elbowRotation: [1.25, -0.3, 0.45],
      wristRotation: [-0.25, 0.1, 0],
      fingerCurl: [0, 0, 0, 0, 0],
    },
    rightArm: {
      shoulderRotation: [-0.55, 0.25, 0.3],
      elbowRotation: [1.25, 0.3, -0.45],
      wristRotation: [-0.25, -0.1, 0],
      fingerCurl: [0, 0, 0, 0, 0],
    },
  },
  HOW: {
    headRotation: [0.06, 0, 0],
    eyebrowFurrow: 0.8,
    leftArm: {
      shoulderRotation: [-0.7, -0.1, -0.2],
      elbowRotation: [1.7, 0, 0.2],
      wristRotation: [-0.2, -0.3, 0],
      fingerCurl: [0.5, 0.5, 0.5, 0.5, 0.5],
    },
    rightArm: {
      shoulderRotation: [-0.7, 0.1, 0.2],
      elbowRotation: [1.7, 0, -0.2],
      wristRotation: [-0.2, 0.3, 0],
      fingerCurl: [0.5, 0.5, 0.5, 0.5, 0.5],
    },
  },
  ME: {
    headRotation: [0.04, 0, 0],
    rightArm: {
      shoulderRotation: [-0.65, -0.32, 0.2],
      elbowRotation: [1.95, -0.4, 0],
      wristRotation: [0.2, 0, 0],
      fingerCurl: [0.9, 0, 0.9, 0.9, 0.9],
    },
  },
  YOU: {
    headRotation: [0.02, 0, 0],
    eyebrowRaise: 0.3,
    rightArm: {
      shoulderRotation: [-0.85, 0.1, 0.1],
      elbowRotation: [1.05, 0, 0],
      wristRotation: [0, 0, 0],
      fingerCurl: [0.9, 0, 0.9, 0.9, 0.9],
    },
  },
  NAME: {
    headRotation: [0.05, 0, 0],
    leftArm: {
      shoulderRotation: [-0.7, -0.18, -0.15],
      elbowRotation: [1.65, 0, 0],
      wristRotation: [0, 0, 0],
      fingerCurl: [1, 0, 0, 1, 1],
    },
    rightArm: {
      shoulderRotation: [-0.82, 0.12, 0.2],
      elbowRotation: [1.82, 0.2, -0.18],
      wristRotation: [0.1, 0, 0],
      fingerCurl: [1, 0, 0, 1, 1],
    },
  },
  GOOD: {
    headRotation: [0.06, 0, 0],
    eyebrowRaise: 0.5,
    mouthSmile: 0.4,
    leftArm: {
      shoulderRotation: [-0.55, -0.2, -0.12],
      elbowRotation: [1.35, 0, 0],
      wristRotation: [0, 0, 0],
      fingerCurl: [0, 0, 0, 0, 0],
    },
    rightArm: {
      shoulderRotation: [-1.05, 0.1, 0.22],
      elbowRotation: [2.05, 0, -0.2],
      wristRotation: [-0.15, 0, 0],
      fingerCurl: [0, 0, 0, 0, 0],
    },
  },
  HELP: {
    headRotation: [0.05, 0, 0],
    eyebrowRaise: 0.5,
    leftArm: {
      shoulderRotation: [-0.62, -0.2, -0.15],
      elbowRotation: [1.4, 0, 0],
      wristRotation: [0, 0, 0],
      fingerCurl: [0, 0, 0, 0, 0],
    },
    rightArm: {
      shoulderRotation: [-0.75, 0.15, 0.2],
      elbowRotation: [1.65, 0, 0],
      wristRotation: [0, 0, 0],
      fingerCurl: [0, 1, 1, 1, 1],
    },
  },
  MEETING: {
    headRotation: [0.04, 0, 0],
    eyebrowRaise: 0.4,
    leftArm: {
      shoulderRotation: [-0.7, -0.22, -0.2],
      elbowRotation: [1.5, 0, 0.2],
      wristRotation: [0, 0, 0],
      fingerCurl: [0.3, 0.3, 0.3, 0.3, 0.3],
    },
    rightArm: {
      shoulderRotation: [-0.7, 0.22, 0.2],
      elbowRotation: [1.5, 0, -0.2],
      wristRotation: [0, 0, 0],
      fingerCurl: [0.3, 0.3, 0.3, 0.3, 0.3],
    },
  },
};

function buildTargetPoseForToken(token: SignRepresentationToken): AvatarPose {
  const glossUpper = token.gloss.toUpperCase();
  const basePreset = GLOSS_POSES[glossUpper] ?? {};

  const resolvedRightArm: ArmPose = basePreset.rightArm
    ? { ...basePreset.rightArm }
    : {
        shoulderRotation: [-0.75, 0.2, 0.25],
        elbowRotation: [1.4, 0, 0],
        wristRotation: [0, 0, 0],
        fingerCurl: [0.1, 0.1, 0.1, 0.1, 0.1],
      };

  const resolvedLeftArm: ArmPose = basePreset.leftArm
    ? { ...basePreset.leftArm }
    : { ...REST_ARM_LEFT };

  // Fingerspelled letters carry their handshape on the signing (right) hand.
  if (token.clipId.startsWith("asl_fs_")) {
    const handshape = handshapeForToken(token.gloss);
    if (handshape !== null) {
      resolvedRightArm.fingerCurl = [handshape[0], handshape[1], handshape[2], handshape[3], handshape[4]];
    }
  }

  // Adjust for spatial loci anchors
  if (token.spatialLoci) {
    const { anchor, targetOffset } = token.spatialLoci;
    if (anchor === "forehead") {
      resolvedRightArm.shoulderRotation[0] -= 0.35;
      resolvedRightArm.elbowRotation[0] += 0.35;
    } else if (anchor === "chest") {
      resolvedRightArm.shoulderRotation[0] = -0.7;
      resolvedRightArm.elbowRotation[0] = 1.7;
    } else if (anchor === "left_shoulder") {
      resolvedRightArm.shoulderRotation[1] -= 0.3;
    }

    resolvedRightArm.shoulderRotation[0] += targetOffset.y * 0.5;
    resolvedRightArm.shoulderRotation[1] += targetOffset.x * 0.5;
  }

  // Face driven 1:1 from the token's linguistic non-manual markers
  // (wh-question furrow, yes/no raise, mouth morphemes, head shake/nod/tilt).
  const face = facePoseForToken(token);

  return {
    headRotation: [face.headPitch, face.headYaw, face.headRoll],
    eyebrowRaise: face.eyebrowRaise,
    eyebrowFurrow: face.eyebrowFurrow,
    mouthOpen: face.mouthOpen,
    mouthSmile: face.mouthSmile,
    mouthWidth: face.mouthWidth,
    chestPitch: 0,
    leftArm: resolvedLeftArm,
    rightArm: resolvedRightArm,
  };
}

export function WebglAvatar({
  representation,
  playbackSpeed = 1.0,
  enableIdleMotion = true,
  onPlaybackComplete,
  className,
}: WebglAvatarProps) {
  const mountRef = useRef<HTMLDivElement>(null);

  const [cameraPreset, setCameraPreset] = useState<"front" | "perspective" | "closeup">("front");

  // Status bar DOM element references for 60fps render-free updates
  const indicatorDotRef = useRef<HTMLDivElement>(null);
  const glossLabelRef = useRef<HTMLSpanElement>(null);
  const progressContainerRef = useRef<HTMLDivElement>(null);
  const progressBarRef = useRef<HTMLDivElement>(null);
  const progressTextRef = useRef<HTMLSpanElement>(null);

  // Three.js internal references
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  // Skeletal hierarchy node references
  const headGroupRef = useRef<THREE.Group | null>(null);
  const leftEyebrowRef = useRef<THREE.Mesh | null>(null);
  const rightEyebrowRef = useRef<THREE.Mesh | null>(null);
  const mouthMeshRef = useRef<THREE.Mesh | null>(null);
  const chestGroupRef = useRef<THREE.Group | null>(null);

  const leftShoulderRef = useRef<THREE.Group | null>(null);
  const leftElbowRef = useRef<THREE.Group | null>(null);
  const leftWristRef = useRef<THREE.Group | null>(null);
  const leftFingerJointsRef = useRef<THREE.Group[]>([]);

  const rightShoulderRef = useRef<THREE.Group | null>(null);
  const rightElbowRef = useRef<THREE.Group | null>(null);
  const rightWristRef = useRef<THREE.Group | null>(null);
  const rightFingerJointsRef = useRef<THREE.Group[]>([]);

  // Animation timeline state
  const currentTokensRef = useRef<SignRepresentationToken[]>([]);
  const currentTokenIndexRef = useRef<number>(-1);
  const tokenStartTimeRef = useRef<number>(0);
  const previousPoseRef = useRef<AvatarPose>(REST_POSE);
  const currentPoseRef = useRef<AvatarPose>(REST_POSE);
  const isPlayingRef = useRef<boolean>(false);
  const onCompleteRef = useRef(onPlaybackComplete);

  useEffect(() => {
    onCompleteRef.current = onPlaybackComplete;
  }, [onPlaybackComplete]);

  // Load new token queue when representation changes
  useEffect(() => {
    if (representation && representation.tokens.length > 0) {
      currentTokensRef.current = [...representation.tokens];
      currentTokenIndexRef.current = 0;
      tokenStartTimeRef.current = performance.now();
      previousPoseRef.current = { ...currentPoseRef.current };
      isPlayingRef.current = true;
      if (glossLabelRef.current) {
        glossLabelRef.current.textContent = representation.tokens[0]?.gloss ?? "REST";
      }
      if (indicatorDotRef.current) {
        indicatorDotRef.current.className = `${styles.signIndicatorDot} ${styles.signIndicatorDotActive}`;
      }
      if (progressContainerRef.current) {
        progressContainerRef.current.style.display = "flex";
      }
    } else {
      currentTokensRef.current = [];
      currentTokenIndexRef.current = -1;
      isPlayingRef.current = false;
      if (glossLabelRef.current) {
        glossLabelRef.current.textContent = "REST";
      }
      if (indicatorDotRef.current) {
        indicatorDotRef.current.className = styles.signIndicatorDot ?? "";
      }
      if (progressContainerRef.current) {
        progressContainerRef.current.style.display = "none";
      }
    }
  }, [representation]);

  // Setup Three.js scene and character geometry
  useEffect(() => {
    const container = mountRef.current;
    if (!container) {
      return;
    }

    const width = container.clientWidth || 400;
    const height = container.clientHeight || 380;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(36, width / height, 0.1, 100);
    camera.position.set(0, 1.42, 2.35);
    camera.lookAt(0, 1.35, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Studio Lighting setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.25);
    keyLight.position.set(1.5, 3.0, 2.5);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x60a5fa, 0.7);
    rimLight.position.set(-2.0, 2.2, -1.8);
    scene.add(rimLight);

    const fillLight = new THREE.PointLight(0x38bdf8, 0.45, 6);
    fillLight.position.set(0, 1.2, 1.2);
    scene.add(fillLight);

    // Procedural Character Materials
    const skinMaterial = new THREE.MeshStandardMaterial({
      color: 0xe2e8f0,
      roughness: 0.45,
      metalness: 0.1,
    });

    const torsoMaterial = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.6,
      metalness: 0.2,
    });

    const jointMaterial = new THREE.MeshStandardMaterial({
      color: 0x2563eb,
      roughness: 0.35,
      metalness: 0.3,
    });

    const facialDetailMaterial = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.3,
    });

    // Root avatar skeleton group
    const avatarRoot = new THREE.Group();
    scene.add(avatarRoot);

    // Pelvis and Spine Base
    const spineBase = new THREE.Group();
    spineBase.position.set(0, 0.88, 0);
    avatarRoot.add(spineBase);

    // Torso Upper Body Mesh
    const torsoGeometry = new THREE.CylinderGeometry(0.24, 0.19, 0.48, 20);
    const torsoMesh = new THREE.Mesh(torsoGeometry, torsoMaterial);
    torsoMesh.position.set(0, 0.24, 0);
    spineBase.add(torsoMesh);

    // Chest Group (Pivot for neck & shoulders)
    const chestGroup = new THREE.Group();
    chestGroup.position.set(0, 0.48, 0);
    spineBase.add(chestGroup);
    chestGroupRef.current = chestGroup;

    // Neck
    const neckGeometry = new THREE.CylinderGeometry(0.08, 0.09, 0.14, 16);
    const neckMesh = new THREE.Mesh(neckGeometry, skinMaterial);
    neckMesh.position.set(0, 0.07, 0);
    chestGroup.add(neckMesh);

    // Head Group
    const headGroup = new THREE.Group();
    headGroup.position.set(0, 0.19, 0);
    chestGroup.add(headGroup);
    headGroupRef.current = headGroup;

    // Stylized Head Mesh
    const headGeometry = new THREE.SphereGeometry(0.165, 32, 32);
    headGeometry.scale(1, 1.15, 1);
    const headMesh = new THREE.Mesh(headGeometry, skinMaterial);
    headGroup.add(headMesh);

    // Left and Right Eyes
    const eyeGeometry = new THREE.SphereGeometry(0.022, 16, 16);
    const leftEye = new THREE.Mesh(eyeGeometry, facialDetailMaterial);
    leftEye.position.set(-0.058, 0.025, 0.142);
    headGroup.add(leftEye);

    const rightEye = new THREE.Mesh(eyeGeometry, facialDetailMaterial);
    rightEye.position.set(0.058, 0.025, 0.142);
    headGroup.add(rightEye);

    // Left and Right Animated Eyebrows
    const eyebrowGeometry = new THREE.BoxGeometry(0.048, 0.012, 0.012);
    const leftEyebrow = new THREE.Mesh(eyebrowGeometry, facialDetailMaterial);
    leftEyebrow.position.set(-0.058, 0.065, 0.15);
    headGroup.add(leftEyebrow);
    leftEyebrowRef.current = leftEyebrow;

    const rightEyebrow = new THREE.Mesh(eyebrowGeometry, facialDetailMaterial);
    rightEyebrow.position.set(0.058, 0.065, 0.15);
    headGroup.add(rightEyebrow);
    rightEyebrowRef.current = rightEyebrow;

    // Animated Mouth Mesh
    const mouthGeometry = new THREE.BoxGeometry(0.06, 0.015, 0.015);
    const mouthMesh = new THREE.Mesh(mouthGeometry, facialDetailMaterial);
    mouthMesh.position.set(0, -0.075, 0.148);
    headGroup.add(mouthMesh);
    mouthMeshRef.current = mouthMesh;

    // Helper: Build Arm Assembly with Hand and 5 Fingers
    function buildArmChain(isLeft: boolean): {
      shoulderGroup: THREE.Group;
      elbowGroup: THREE.Group;
      wristGroup: THREE.Group;
      fingerJoints: THREE.Group[];
    } {
      const sideMultiplier = isLeft ? -1 : 1;

      // Shoulder Pivot
      const shoulderGroup = new THREE.Group();
      shoulderGroup.position.set(0.24 * sideMultiplier, -0.02, 0);
      chestGroup.add(shoulderGroup);

      const shoulderJointMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.065, 16, 16),
        jointMaterial,
      );
      shoulderGroup.add(shoulderJointMesh);

      // Upper Arm Segment
      const upperArmMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.05, 0.045, 0.28, 16),
        skinMaterial,
      );
      upperArmMesh.position.set(0, -0.14, 0);
      shoulderGroup.add(upperArmMesh);

      // Elbow Pivot
      const elbowGroup = new THREE.Group();
      elbowGroup.position.set(0, -0.28, 0);
      shoulderGroup.add(elbowGroup);

      const elbowJointMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.05, 16, 16),
        jointMaterial,
      );
      elbowGroup.add(elbowJointMesh);

      // Forearm Segment
      const forearmMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.045, 0.038, 0.26, 16),
        skinMaterial,
      );
      forearmMesh.position.set(0, -0.13, 0);
      elbowGroup.add(forearmMesh);

      // Wrist Pivot
      const wristGroup = new THREE.Group();
      wristGroup.position.set(0, -0.26, 0);
      elbowGroup.add(wristGroup);

      const wristJointMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.04, 16, 16),
        jointMaterial,
      );
      wristGroup.add(wristJointMesh);

      // Palm Mesh
      const palmMesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.068, 0.075, 0.024),
        skinMaterial,
      );
      palmMesh.position.set(0, -0.045, 0);
      wristGroup.add(palmMesh);

      // 5 Fingers: Thumb, Index, Middle, Ring, Pinky
      const fingerJoints: THREE.Group[] = [];
      const fingerOffsets = [
        { x: -0.04 * sideMultiplier, y: -0.02, z: 0.015, length: 0.035, radius: 0.01 }, // Thumb
        { x: -0.025, y: -0.082, z: 0, length: 0.045, radius: 0.009 },                     // Index
        { x: -0.008, y: -0.088, z: 0, length: 0.05, radius: 0.009 },                      // Middle
        { x: 0.01, y: -0.082, z: 0, length: 0.044, radius: 0.0085 },                     // Ring
        { x: 0.026, y: -0.075, z: 0, length: 0.036, radius: 0.008 },                      // Pinky
      ];

      for (let i = 0; i < 5; i++) {
        const offset = fingerOffsets[i];
        if (!offset) {
          continue;
        }

        const fingerBase = new THREE.Group();
        fingerBase.position.set(offset.x, offset.y, offset.z);
        wristGroup.add(fingerBase);

        const phalanxMesh = new THREE.Mesh(
          new THREE.CylinderGeometry(offset.radius, offset.radius * 0.85, offset.length, 10),
          skinMaterial,
        );
        phalanxMesh.position.set(0, -offset.length * 0.5, 0);
        fingerBase.add(phalanxMesh);

        fingerJoints.push(fingerBase);
      }

      return { shoulderGroup, elbowGroup, wristGroup, fingerJoints };
    }

    const leftArmChain = buildArmChain(true);
    leftShoulderRef.current = leftArmChain.shoulderGroup;
    leftElbowRef.current = leftArmChain.elbowGroup;
    leftWristRef.current = leftArmChain.wristGroup;
    leftFingerJointsRef.current = leftArmChain.fingerJoints;

    const rightArmChain = buildArmChain(false);
    rightShoulderRef.current = rightArmChain.shoulderGroup;
    rightElbowRef.current = rightArmChain.elbowGroup;
    rightWristRef.current = rightArmChain.wristGroup;
    rightFingerJointsRef.current = rightArmChain.fingerJoints;

    // Responsive Canvas Resize Observer
    const handleResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) {
        return;
      }
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;
      cameraRef.current.aspect = newWidth / newHeight;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(newWidth, newHeight);
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    // Animation Loop
    let animationFrameId: number;

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const now = performance.now();

      // Sign Representation playback timeline
      if (isPlayingRef.current && currentTokensRef.current.length > 0) {
        const tokens = currentTokensRef.current;
        const index = currentTokenIndexRef.current;
        const currentToken = tokens[index];

        if (currentToken) {
          const timing = currentToken.timing;
          const leadInMs = timing.leadInDurationMs / playbackSpeed;
          const holdMs = timing.holdDurationMs / playbackSpeed;
          const leadOutMs = timing.leadOutDurationMs / playbackSpeed;
          const totalTokenDuration = leadInMs + holdMs + leadOutMs;

          const elapsedMs = now - tokenStartTimeRef.current;
          const clampedElapsed = Math.min(elapsedMs, totalTokenDuration);
          const progressPercent = Math.min(100, Math.round((clampedElapsed / totalTokenDuration) * 100));
          if (progressBarRef.current) {
            progressBarRef.current.style.width = `${progressPercent}%`;
          }
          if (progressTextRef.current) {
            progressTextRef.current.textContent = `${progressPercent}%`;
          }

          const targetPose = buildTargetPoseForToken(currentToken);

          if (elapsedMs < leadInMs) {
            // Phase 1: Lead-In Transition (coarticulated from end of previous sign)
            const rawT = leadInMs > 0 ? elapsedMs / leadInMs : 1;
            currentPoseRef.current = blendPoses(previousPoseRef.current, targetPose, rawT);
          } else if (elapsedMs < leadInMs + holdMs) {
            // Phase 2: Hold Sign Stroke
            currentPoseRef.current = targetPose;
          } else if (elapsedMs < totalTokenDuration) {
            // Phase 3: Lead-Out Transition (coarticulated into the next sign)
            const nextToken = tokens[index + 1];
            const nextTargetPose = nextToken ? buildTargetPoseForToken(nextToken) : REST_POSE;
            const outElapsed = elapsedMs - leadInMs - holdMs;
            const rawT = leadOutMs > 0 ? outElapsed / leadOutMs : 1;
            currentPoseRef.current = blendPoses(targetPose, nextTargetPose, rawT);
          } else {
            // Advance to next token or finalize
            if (index + 1 < tokens.length) {
              currentTokenIndexRef.current = index + 1;
              tokenStartTimeRef.current = now;
              previousPoseRef.current = { ...currentPoseRef.current };
              const nextGloss = tokens[index + 1]?.gloss ?? "REST";
              if (glossLabelRef.current) {
                glossLabelRef.current.textContent = nextGloss;
              }
            } else {
              isPlayingRef.current = false;
              currentTokenIndexRef.current = -1;
              currentPoseRef.current = REST_POSE;
              if (glossLabelRef.current) {
                glossLabelRef.current.textContent = "REST";
              }
              if (indicatorDotRef.current) {
                indicatorDotRef.current.className = styles.signIndicatorDot ?? "";
              }
              if (progressContainerRef.current) {
                progressContainerRef.current.style.display = "none";
              }
              if (onCompleteRef.current) {
                onCompleteRef.current();
              }
            }
          }
        }
      } else if (enableIdleMotion) {
        // Natural idle breathing and subtle drift
        const idleTime = now * 0.0018;
        const breath = Math.sin(idleTime * 1.8) * 0.02;
        const microHead = Math.sin(idleTime * 0.8) * 0.015;

        const idlePose: AvatarPose = {
          ...REST_POSE,
          chestPitch: breath,
          headRotation: [microHead, Math.sin(idleTime * 0.5) * 0.015, 0],
        };

        // Smooth return to idle rest pose
        currentPoseRef.current = blendPoses(currentPoseRef.current, idlePose, 0.08);
      }

      // Apply currentPoseRef to Three.js Skeleton Transforms
      const pose = currentPoseRef.current;

      if (chestGroupRef.current) {
        chestGroupRef.current.rotation.x = pose.chestPitch;
      }

      if (headGroupRef.current) {
        headGroupRef.current.rotation.set(
          pose.headRotation[0],
          pose.headRotation[1],
          pose.headRotation[2],
        );
      }

      if (leftEyebrowRef.current) {
        const eyebrowY = 0.065 + pose.eyebrowRaise * 0.018 - pose.eyebrowFurrow * 0.01;
        leftEyebrowRef.current.position.y = eyebrowY;
        leftEyebrowRef.current.rotation.z = -pose.eyebrowFurrow * 0.25;
      }

      if (rightEyebrowRef.current) {
        const eyebrowY = 0.065 + pose.eyebrowRaise * 0.018 - pose.eyebrowFurrow * 0.01;
        rightEyebrowRef.current.position.y = eyebrowY;
        rightEyebrowRef.current.rotation.z = pose.eyebrowFurrow * 0.25;
      }

      if (mouthMeshRef.current) {
        mouthMeshRef.current.scale.set(
          pose.mouthWidth,
          1 + pose.mouthOpen * 2.5,
          1,
        );
        mouthMeshRef.current.position.y = -0.075 - pose.mouthOpen * 0.015;
      }

      // Left Arm Joint Transforms
      if (leftShoulderRef.current) {
        leftShoulderRef.current.rotation.set(
          pose.leftArm.shoulderRotation[0],
          pose.leftArm.shoulderRotation[1],
          pose.leftArm.shoulderRotation[2],
        );
      }
      if (leftElbowRef.current) {
        leftElbowRef.current.rotation.set(
          pose.leftArm.elbowRotation[0],
          pose.leftArm.elbowRotation[1],
          pose.leftArm.elbowRotation[2],
        );
      }
      if (leftWristRef.current) {
        leftWristRef.current.rotation.set(
          pose.leftArm.wristRotation[0],
          pose.leftArm.wristRotation[1],
          pose.leftArm.wristRotation[2],
        );
      }
      for (let i = 0; i < leftFingerJointsRef.current.length; i++) {
        const finger = leftFingerJointsRef.current[i];
        const curl = pose.leftArm.fingerCurl[i] ?? 0;
        if (finger) {
          finger.rotation.x = curl * 1.5;
        }
      }

      // Right Arm Joint Transforms
      if (rightShoulderRef.current) {
        rightShoulderRef.current.rotation.set(
          pose.rightArm.shoulderRotation[0],
          pose.rightArm.shoulderRotation[1],
          pose.rightArm.shoulderRotation[2],
        );
      }
      if (rightElbowRef.current) {
        rightElbowRef.current.rotation.set(
          pose.rightArm.elbowRotation[0],
          pose.rightArm.elbowRotation[1],
          pose.rightArm.elbowRotation[2],
        );
      }
      if (rightWristRef.current) {
        rightWristRef.current.rotation.set(
          pose.rightArm.wristRotation[0],
          pose.rightArm.wristRotation[1],
          pose.rightArm.wristRotation[2],
        );
      }
      for (let i = 0; i < rightFingerJointsRef.current.length; i++) {
        const finger = rightFingerJointsRef.current[i];
        const curl = pose.rightArm.fingerCurl[i] ?? 0;
        if (finger) {
          finger.rotation.x = curl * 1.5;
        }
      }

      // Render Three.js frame
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      if (rendererRef.current && rendererRef.current.domElement) {
        rendererRef.current.dispose();
        if (rendererRef.current.domElement.parentElement) {
          rendererRef.current.domElement.parentElement.removeChild(
            rendererRef.current.domElement,
          );
        }
      }
    };
  }, [enableIdleMotion, playbackSpeed]);

  // Adjust camera preset positions smoothly
  const handleCameraChange = useCallback((preset: "front" | "perspective" | "closeup") => {
    setCameraPreset(preset);
    if (!cameraRef.current) {
      return;
    }
    switch (preset) {
      case "front":
        cameraRef.current.position.set(0, 1.42, 2.35);
        cameraRef.current.lookAt(0, 1.35, 0);
        break;
      case "perspective":
        cameraRef.current.position.set(0.65, 1.46, 2.15);
        cameraRef.current.lookAt(0, 1.35, 0);
        break;
      case "closeup":
        cameraRef.current.position.set(0, 1.52, 1.35);
        cameraRef.current.lookAt(0, 1.52, 0);
        break;
    }
  }, []);

  return (
    <div
      className={`${styles.avatarContainer} ${className ?? ""}`}
      role="region"
      aria-label="ASL Skeletal WebGL 3D Avatar"
    >
      {/* View Presets */}
      <div className={styles.overlayControls}>
        <button
          type="button"
          onClick={() => handleCameraChange("front")}
          className={`${styles.controlBtn} ${cameraPreset === "front" ? styles.controlBtnActive : ""}`}
          aria-label="Front View"
        >
          Front
        </button>
        <button
          type="button"
          onClick={() => handleCameraChange("perspective")}
          className={`${styles.controlBtn} ${cameraPreset === "perspective" ? styles.controlBtnActive : ""}`}
          aria-label="3/4 Perspective View"
        >
          3/4 View
        </button>
        <button
          type="button"
          onClick={() => handleCameraChange("closeup")}
          className={`${styles.controlBtn} ${cameraPreset === "closeup" ? styles.controlBtnActive : ""}`}
          aria-label="Close-up View"
        >
          Close-up
        </button>
      </div>

      {/* WebGL Canvas Container */}
      <div ref={mountRef} className={styles.canvasWrapper} />

      {/* Playback Status Bar */}
      <div className={styles.statusBar}>
        <div className={styles.currentSignBadge}>
          <div ref={indicatorDotRef} className={styles.signIndicatorDot} />
          <span ref={glossLabelRef} className={styles.signGlossLabel}>
            REST
          </span>
        </div>

        <div
          ref={progressContainerRef}
          className={styles.tokenProgressWrapper}
          style={{ display: "none" }}
        >
          <div className={styles.progressBarContainer}>
            <div
              ref={progressBarRef}
              className={styles.progressBarFill}
              style={{ width: "0%" }}
            />
          </div>
          <span ref={progressTextRef} className={styles.tokenMeta}>
            0%
          </span>
        </div>
      </div>
    </div>
  );
}
