import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { clone as cloneSkeleton } from "three/addons/utils/SkeletonUtils.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { DEAL_MS, REVEAL_MS, DECK, dealPose, revealPose } from "./card-motion.js";

const SUITS = { c: "♣", d: "♦", h: "♥", s: "♠" };
const RANKS = { T: "10", J: "J", Q: "Q", K: "K", A: "A" };
const CHIP_VALUES = [25, 5, 1, .5];
const CHIP_COLORS = ["#292b29", "#8a3935", "#d5ceb6", "#356452"];

function canvasTexture(width, height, paint) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  paint(canvas.getContext("2d"), width, height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

function grainTexture(wood) {
  return canvasTexture(512, 512, (ctx, width, height) => {
    const pixels = ctx.createImageData(width, height);
    // A fixed texture keeps the room stable between hands and screenshots.
    let seed = 42;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      const noise = seed / 4294967296;
      const grain = wood ? Math.sin(x * .27 + Math.sin(y * .018) * 3) * 9 : 0;
      const base = wood ? [43, 26, 17] : [9, 49, 35];
      const index = (y * width + x) * 4;
      base.forEach((value, channel) => { pixels.data[index + channel] = value + noise * (wood ? 13 : 19) + grain; });
      pixels.data[index + 3] = 255;
    }
    ctx.putImageData(pixels, 0, 0);
  });
}

function cardTexture(card) {
  return canvasTexture(256, 384, (ctx, width, height) => {
    ctx.fillStyle = "#eeeadd";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#cfc9b8";
    ctx.lineWidth = 2;
    ctx.strokeRect(9, 9, width - 18, height - 18);
    if (!card) {
      ctx.fillStyle = "#233c30";
      ctx.fillRect(15, 15, width - 30, height - 30);
      ctx.save();
      ctx.beginPath();
      ctx.rect(20, 20, width - 40, height - 40);
      ctx.clip();
      ctx.strokeStyle = "#b7bd9b55";
      ctx.lineWidth = 2;
      for (let i = -height; i < width + height; i += 16) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i + height, height); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i - height, height); ctx.stroke();
      }
      ctx.restore();
      ctx.fillStyle = "#233c30";
      ctx.beginPath(); ctx.arc(width / 2, height / 2, 44, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#cfc4a3";
      ctx.font = "50px Georgia";
      ctx.textAlign = "center";
      ctx.fillText("♠", width / 2, height / 2 + 16);
      return;
    }
    const rank = RANKS[card[0]] || card[0], suit = SUITS[card[1]];
    ctx.fillStyle = card[1] === "h" || card[1] === "d" ? "#983b36" : "#202b25";
    ctx.font = "bold 72px Georgia";
    ctx.fillText(rank, 20, 81);
    ctx.font = "48px Georgia";
    ctx.fillText(suit, 22, 131);
    ctx.save();
    ctx.translate(width, height); ctx.rotate(Math.PI);
    ctx.font = "bold 72px Georgia"; ctx.fillText(rank, 20, 81);
    ctx.font = "48px Georgia"; ctx.fillText(suit, 22, 131);
    ctx.restore();
    ctx.textAlign = "center";
    ctx.font = "112px Georgia";
    ctx.fillText(suit, width / 2, height / 2 + 38);
  });
}

function chipTexture(value, color, side = false) {
  return canvasTexture(side ? 512 : 256, side ? 64 : 256, (ctx, w, h) => {
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, w, h);
    const ink = value === 1 ? "#34453b" : "#f4e9cb";
    if (side) {
      for (let i = 0; i < 8; i++) {
        ctx.fillStyle = "#eee5ce";
        ctx.fillRect(i * 64 + 22, 0, 20, h);
        ctx.fillStyle = color;
        ctx.fillRect(i * 64 + 29, 0, 6, h);
      }
      ctx.fillStyle = "#0003";
      ctx.fillRect(0, 0, w, 5); ctx.fillRect(0, h - 5, w, 5);
      return;
    }
    ctx.translate(128, 128);
    ctx.save();
    for (let i = 0; i < 8; i++) {
      ctx.fillStyle = "#eee5ce";
      ctx.fillRect(-19, 96, 38, 34);
      ctx.fillStyle = color;
      ctx.fillRect(-5, 100, 10, 30);
      ctx.rotate(Math.PI / 4);
    }
    ctx.restore();
    ctx.strokeStyle = ink;
    for (const radius of [80, 86, 118]) {
      ctx.lineWidth = radius === 86 ? 2 : 1;
      ctx.beginPath(); ctx.arc(0, 0, radius, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = ink;
    ctx.textAlign = "center";
    ctx.font = "18px Georgia"; ctx.fillText("♠", 0, -42);
    ctx.font = "bold 56px Georgia"; ctx.fillText(String(value), 0, 20);
    ctx.font = "14px sans-serif"; ctx.fillText("B B", 0, 50);
  });
}

export async function createTableScene(canvas, host) {
  const loader = new GLTFLoader();
  const handModels = await Promise.all(["left", "right"].map((side) => loader.loadAsync(`/assets/hands/${side}.glb`)));
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "low-power" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#090b0a");
  scene.fog = new THREE.FogExp2("#090b0a", .055);
  const camera = new THREE.PerspectiveCamera(56, 1, .1, 50);
  const foreground = new THREE.Scene();
  const handCamera = new THREE.PerspectiveCamera(56, 1, .1, 10);
  renderer.autoClear = false;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const pointer = new THREE.Vector2();
  let mobile = false, inView = true, frame = null, previousTime = 0, disposed = false;
  const moving = [];
  const ease = (t) => t * t * (3 - 2 * t);
  function animate(duration, sample, done = () => {}, delay = 0, channel = "action") {
    if (reducedMotion.matches) { sample(1); done(); return; }
    moving.push({ start: performance.now() + delay, duration, sample, done, channel });
  }
  function settleAnimations(includeCards) {
    for (let i = moving.length - 1; i >= 0; i--) {
      const motion = moving[i];
      if (!includeCards && motion.channel === "cards") continue;
      moving.splice(i, 1);
      motion.sample(1); motion.done();
    }
  }
  const geometries = new Set(), materials = new Set(), textures = new Set(), skeletons = new Set();
  const geometry = (value) => { geometries.add(value); return value; };
  const material = (value) => { materials.add(value); return value; };
  const texture = (value) => { textures.add(value); return value; };
  const standard = (options) => material(new THREE.MeshStandardMaterial(options));

  function mesh(shape, surface, parent, x = 0, y = 0, z = 0) {
    const object = new THREE.Mesh(shape, surface);
    object.position.set(x, y, z);
    object.castShadow = true;
    object.receiveShadow = true;
    parent.add(object);
    return object;
  }

  scene.add(new THREE.HemisphereLight("#bbd0c3", "#080a08", .13));
  const key = new THREE.SpotLight("#ffe7c4", 125, 15, .61, .6, 2);
  key.position.set(-.8, 4.6, -.45);
  key.target.position.set(0, 0, -.2);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -.00025;
  key.shadow.normalBias = .02;
  scene.add(key, key.target);
  const fill = new THREE.DirectionalLight("#acc9c2", .18);
  fill.position.set(2, 4, 4);
  scene.add(fill);

  const cylinder = geometry(new THREE.CylinderGeometry(1, 1, 1, 128));
  const woodMap = texture(grainTexture(true));
  woodMap.wrapS = woodMap.wrapT = THREE.RepeatWrapping;
  woodMap.repeat.set(3, 2);
  const wood = standard({ map: woodMap, roughness: .38, metalness: .05 });
  const tableBase = mesh(cylinder, wood, scene, 0, -.22, 0);
  tableBase.scale.set(3.85, .36, 2.85);
  const brass = standard({ color: "#857149", metalness: .7, roughness: .35 });
  const trim = mesh(cylinder, brass, scene, 0, -.025, 0);
  trim.scale.set(3.53, .035, 2.56);
  const feltMap = texture(grainTexture(false));
  feltMap.wrapS = feltMap.wrapT = THREE.RepeatWrapping;
  feltMap.repeat.set(5, 4);
  const felt = standard({ map: feltMap, bumpMap: feltMap, bumpScale: .015, roughness: 1 });
  const tableTop = mesh(cylinder, felt, scene, 0, .005, 0);
  tableTop.scale.set(3.49, .045, 2.52);

  // A faint stitched ellipse is part of the felt, below the cards and chips.
  const points = Array.from({ length: 129 }, (_, i) => {
    const angle = i / 128 * Math.PI * 2;
    return new THREE.Vector3(Math.cos(angle) * 3.23, .03, Math.sin(angle) * 2.28);
  });
  const stitch = new THREE.LineLoop(geometry(new THREE.BufferGeometry().setFromPoints(points)), material(new THREE.LineBasicMaterial({ color: "#81956c", transparent: true, opacity: .25 })));
  scene.add(stitch);

  const leatherMap = texture(canvasTexture(256, 256, (ctx, w, h) => {
    ctx.fillStyle = "#777"; ctx.fillRect(0, 0, w, h);
    for (let y = 0; y < h; y += 3) for (let x = 0; x < w; x += 3) {
      ctx.fillStyle = (x * 7 + y * 13) % 5 ? "#888" : "#555";
      ctx.fillRect(x, y, 1, 2);
    }
  }));
  leatherMap.wrapS = leatherMap.wrapT = THREE.RepeatWrapping;
  leatherMap.repeat.set(6, 6);
  const glove = standard({ color: "#f4f1e7", roughness: .76, bumpMap: leatherMap, bumpScale: .002 });
  const sleeve = standard({ color: "#0e1315", roughness: .95 });
  const cuffGeometry = geometry(new THREE.CylinderGeometry(.095, .15, 1.5, 32));
  const handBasis = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, -1, 0), new THREE.Vector3(1, 0, 0),
  );
  function makeHand(side, curl = .3) {
    const group = new THREE.Group();
    const model = cloneSkeleton(handModels[side === "left" ? 0 : 1].scene);
    const wrist = model.getObjectByName("wrist");
    const origin = wrist.position.clone();
    model.updateMatrixWorld(true);
    // WebXR joints are flat. Reparent without changing their bind transforms so
    // finger bends propagate naturally through the rig.
    for (const finger of ["thumb", "index-finger", "middle-finger", "ring-finger", "pinky-finger"]) {
      const names = finger === "thumb"
        ? ["metacarpal", "phalanx-proximal", "phalanx-distal", "tip"]
        : ["metacarpal", "phalanx-proximal", "phalanx-intermediate", "phalanx-distal", "tip"];
      let parent = wrist;
      names.forEach((part) => {
        const bone = model.getObjectByName(`${finger}-${part}`);
        parent.attach(bone);
        parent = bone;
      });
      for (const part of names.slice(1, -1)) {
        model.getObjectByName(`${finger}-${part}`).rotateX(finger === "thumb" ? -curl * .3 : -curl);
      }
    }
    model.position.sub(origin);
    model.traverse((node) => {
      if (!node.isMesh) return;
      geometries.add(node.geometry);
      node.material = glove;
      node.castShadow = true; node.receiveShadow = true;
      node.frustumCulled = false;
      if (node.skeleton) skeletons.add(node.skeleton);
    });
    const normalized = new THREE.Group();
    normalized.setRotationFromMatrix(handBasis);
    if (side === "left") normalized.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), Math.PI);
    normalized.scale.setScalar(3.7);
    normalized.add(model);
    group.add(normalized);
    mesh(cuffGeometry, sleeve, group, 0, -.70, 0);
    return group;
  }
  const opponentHands = [makeHand("right"), makeHand("left")];
  opponentHands.forEach((hand, i) => {
    hand.position.set(i ? 1.2 : -1.2, 0, -2.38);
    hand.rotation.set(Math.PI / 2, Math.PI, i ? .18 : -.18);
    scene.add(hand);
  });
  const bettingHand = makeHand("left", .3);
  bettingHand.position.set(-1.9, 0, 2.6);
  bettingHand.rotation.set(Math.PI / 2, Math.PI, Math.PI - .7);
  scene.add(bettingHand);
  const gestureHands = [...opponentHands, bettingHand];
  gestureHands.forEach((hand) => {
    // Measure the posed skin, including fingertips and sleeves, rather than the
    // bind-pose bounds. All gestures lift from this clearance above the felt.
    hand.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(hand, true);
    hand.position.y += .055 - bounds.min.y;
    hand.userData.rest = hand.position.clone();
  });

  // A separate foreground pass prevents the table and chip stacks from cutting
  // through your held cards. The two cards have distinct, parallel depth planes.
  const pocket = new THREE.Group();
  foreground.add(pocket);
  const pocketPose = new THREE.Group();
  pocket.add(pocketPose);
  const heldCards = new THREE.Group();
  pocketPose.add(heldCards);
  // Tuck the fingers below the card edge, with the thumb in front of the face.
  const grip = makeHand("right", 1.2);
  grip.position.set(.32, -.76, .32);
  grip.rotation.set(0, .10, -.30);
  pocketPose.add(grip);
  foreground.add(new THREE.HemisphereLight("#e1e9e3", "#30372e", 1.6));
  const nearLight = new THREE.DirectionalLight("#ffefda", 2.3);
  nearLight.position.set(-1, 2, 3);
  foreground.add(nearLight);

  const cardShape = new THREE.Shape();
  const width = .57, height = .84, radius = .035;
  cardShape.moveTo(-width / 2 + radius, -height / 2);
  cardShape.lineTo(width / 2 - radius, -height / 2);
  cardShape.quadraticCurveTo(width / 2, -height / 2, width / 2, -height / 2 + radius);
  cardShape.lineTo(width / 2, height / 2 - radius);
  cardShape.quadraticCurveTo(width / 2, height / 2, width / 2 - radius, height / 2);
  cardShape.lineTo(-width / 2 + radius, height / 2);
  cardShape.quadraticCurveTo(-width / 2, height / 2, -width / 2, height / 2 - radius);
  cardShape.lineTo(-width / 2, -height / 2 + radius);
  cardShape.quadraticCurveTo(-width / 2, -height / 2, -width / 2 + radius, -height / 2);
  const cardGeometry = geometry(new THREE.ExtrudeGeometry(cardShape, { depth: .008, bevelEnabled: false, curveSegments: 5 }));
  cardGeometry.rotateX(-Math.PI / 2);
  const positions = cardGeometry.getAttribute("position");
  const uv = cardGeometry.getAttribute("uv");
  for (let i = 0; i < positions.count; i++) {
    uv.setXY(i, (positions.getX(i) + width / 2) / width, (-positions.getZ(i) + height / 2) / height);
  }
  const normals = cardGeometry.getAttribute("normal");
  cardGeometry.clearGroups();
  let runStart = 0, runMaterial = -1;
  for (let i = 0; i < normals.count; i += 3) {
    const side = normals.getY(i) > .9 ? 0 : normals.getY(i) < -.9 ? 1 : 2;
    if (side !== runMaterial) {
      if (i) cardGeometry.addGroup(runStart, i - runStart, runMaterial);
      runStart = i; runMaterial = side;
    }
  }
  cardGeometry.addGroup(runStart, normals.count - runStart, runMaterial);
  const paperEdge = standard({ color: "#ddd8c9", roughness: .8 });
  const cardMaterials = new Map();
  function getCardMaterial(card) {
    if (!cardMaterials.has(card)) cardMaterials.set(card, standard({ map: texture(cardTexture(card)), roughness: .73 }));
    return [cardMaterials.get(card), cardMaterials.get(null), paperEdge];
  }
  getCardMaterial(null);
  const cards = new THREE.Group();
  scene.add(cards);
  const cardMeshes = new Map();
  let cardSignature = "", renderedHand = null;
  function applyCardPose(object, pose) {
    object.position.set(pose.x, pose.y, pose.z);
    object.rotation.set(0, pose.yaw, pose.roll);
    object.scale.setScalar(pose.scale);
  }
  function placeCard(slot, card, x, z, rotation = 0, delay = 0, held = false) {
    const now = performance.now();
    if (cardMeshes.has(slot)) {
      const object = cardMeshes.get(slot);
      if (object.userData.card !== card) {
        // The face is on the underside until the physical half-turn reveals it.
        // Wait for an in-flight deal instead of snapping that card to its seat.
        object.material = getCardMaterial(card);
        const wait = Math.max(delay, object.userData.readyAt - now);
        animate(REVEAL_MS, (t) => applyCardPose(object, revealPose(t, object.userData.target)), undefined, wait, "cards");
        object.userData.readyAt = reducedMotion.matches ? now : now + wait + REVEAL_MS;
        object.userData.card = card;
      }
      return object.userData.readyAt;
    }
    const object = mesh(cardGeometry, getCardMaterial(card), held ? heldCards : cards, x, held ? .02 : .042, z);
    cardMeshes.set(slot, object);
    object.userData.card = card;
    object.userData.readyAt = now;
    if (held) {
      object.rotation.order = "ZYX";
      object.rotation.set(Math.PI / 2, 0, rotation);
      object.castShadow = false;
      object.receiveShadow = false;
    } else {
      const target = { x, z, yaw: rotation, board: slot.startsWith("board") };
      object.userData.target = target;
      if (card && !target.board) {
        applyCardPose(object, revealPose(1, target));
      } else {
        applyCardPose(object, dealPose(0, target));
        object.visible = reducedMotion.matches;
        animate(DEAL_MS, (t) => {
          object.visible = true;
          applyCardPose(object, dealPose(t, target));
        }, undefined, delay, "cards");
        object.userData.readyAt = reducedMotion.matches ? now : now + delay + DEAL_MS;
      }
    }
    return object.userData.readyAt;
  }
  // A shared, visible origin makes each deal read as a card leaving the pack.
  for (let i = 0; i < 7; i++) {
    const card = mesh(cardGeometry, getCardMaterial(null), scene, DECK.x, .042 + i * .009, DECK.z);
    card.rotation.y = .16;
  }

  const chipRim = new THREE.LatheGeometry([
    new THREE.Vector2(.102, -.016), new THREE.Vector2(.108, -.010),
    new THREE.Vector2(.108, .010), new THREE.Vector2(.102, .016),
  ], 48);
  const chipFace = new THREE.CircleGeometry(.102, 48).rotateX(-Math.PI / 2).translate(0, .016, 0);
  const chipBack = chipFace.clone().rotateX(Math.PI);
  const chipGeometry = geometry(mergeGeometries([chipRim, chipFace, chipBack], true));
  [chipRim, chipFace, chipBack].forEach((part) => part.dispose());
  const chipMaterials = CHIP_VALUES.map((value, i) => {
    const side = standard({ map: texture(chipTexture(value, CHIP_COLORS[i], true)), roughness: .58 });
    const top = standard({ map: texture(chipTexture(value, CHIP_COLORS[i])), roughness: .68 });
    return [side, top, top];
  });
  const chips = new THREE.Group();
  scene.add(chips);
  let chipSignature = "";
  function placeChips(amount, x, z, reserve = false) {
    const pile = new THREE.Group();
    pile.position.set(x, 0, z);
    chips.add(pile);
    let halfUnits = Math.round(amount * 2);
    const counts = [0, 0, 0, 0];
    // Reserve stacks use smaller chips too; the displayed value stays exact.
    if (reserve && halfUnits >= 40) {
      counts[2] = 10; halfUnits -= 20;
      counts[1] = Math.min(4, Math.floor(halfUnits / 10)); halfUnits -= counts[1] * 10;
    }
    CHIP_VALUES.forEach((value, i) => {
      const units = value * 2;
      const count = Math.floor(halfUnits / units);
      counts[i] += count;
      halfUnits -= count * units;
    });
    let column = 0;
    counts.forEach((count, denomination) => {
      for (let i = 0; i < count; i++) {
        const stack = Math.floor(i / 12);
        const object = mesh(chipGeometry, chipMaterials[denomination], pile,
          (column + stack) * .245 + Math.sin(i * 8) * .004, .048 + (i % 12) * .034, (stack % 2) * .035);
        object.rotation.y = i * .42;
      }
      if (count) column += Math.ceil(count / 12);
    });
    return pile;
  }
  const reservePoints = [new THREE.Vector3(-1.6, 0, 1.78), new THREE.Vector3(-1.95, 0, -1.83)];
  const betPoints = [new THREE.Vector3(-.65, 0, 1.48), new THREE.Vector3(-.28, 0, -1.14)];
  const potPoint = new THREE.Vector3(-1.73, 0, -.7);
  function drawChips(state, previous = null, revealDelay = 0) {
    chips.clear();
    const players = [state.user, state.agent];
    const paid = players.map((player) => previous ? Math.max(0, state.committed[player] - previous.committed[player]) : 0);
    const collect = previous && state.pot > previous.pot;
    const payout = previous && state.finished && !previous.finished;
    const active = previous && !reducedMotion.matches && (paid.some(Boolean) || collect || payout);
    const awards = players.map((player) => payout ? state.winner === null ? state.pot / 2 : state.winner === player ? state.pot : 0 : 0);
    players.forEach((player, i) => {
      const point = reservePoints[i];
      placeChips(state.stacks[player] - (active ? awards[i] : 0), point.x, point.z, true);
    });
    if (!active) {
      players.forEach((player, i) => placeChips(state.bets[player], betPoints[i].x, betPoints[i].z));
      if (!state.finished) placeChips(state.pot, potPoint.x, potPoint.z);
      return;
    }
    const oldPot = placeChips(collect ? previous.pot : state.pot, potPoint.x, potPoint.z);
    // A called all-in may return an unmatched excess. Only move the matched
    // contributions into the pot; a fold keeps the unequal bets as posted.
    const folded = state.histories.flat().at(-1) === "fold";
    const incoming = players.map((player, i) => {
      const amount = collect ? folded ? previous.bets[player] : (state.pot - previous.pot) / 2 : state.bets[player];
      return placeChips(amount, betPoints[i].x, betPoints[i].z);
    });
    const finalPot = placeChips(state.pot, potPoint.x, potPoint.z);
    finalPot.visible = false;
    const winnings = awards.map((amount) => placeChips(amount, potPoint.x, potPoint.z));
    winnings.forEach((pile) => { pile.visible = false; });
    const pushEnd = paid.some(Boolean) ? 380 : 0;
    const collectEnd = pushEnd + (collect ? 380 : 0);
    const payoutStart = Math.max(collectEnd, revealDelay);
    const duration = (payout ? payoutStart + 680 : collectEnd) || 380;
    const settled = { ...state, stacks: [...state.stacks], bets: [...state.bets] };
    animate(duration, (t) => {
      const elapsed = t * duration;
      incoming.forEach((pile, i) => {
        pile.position.copy(betPoints[i]);
        if (paid[i] && elapsed < pushEnd) {
          const progress = ease(Math.min(1, elapsed / pushEnd));
          pile.position.lerpVectors(reservePoints[i], betPoints[i], progress);
          pile.position.y = Math.sin(progress * Math.PI) * .025;
        } else if (collect) {
          pile.position.lerpVectors(betPoints[i], potPoint, ease(Math.min(1, (elapsed - pushEnd) / 380)));
        }
        pile.visible = !collect || elapsed < collectEnd;
      });
      if (collect) oldPot.visible = elapsed < collectEnd;
      finalPot.visible = (collect || payout) && elapsed >= collectEnd && (!payout || elapsed < payoutStart);
      if (payout) {
        oldPot.visible = elapsed < collectEnd;
        winnings.forEach((pile, i) => {
          pile.visible = elapsed >= payoutStart && awards[i] > 0;
          pile.position.lerpVectors(potPoint, reservePoints[i], ease(Math.max(0, Math.min(1, (elapsed - payoutStart - 120) / 560))));
        });
      }
    }, () => drawChips(settled));
  }

  function gesture(action, player, user) {
    const hand = player === user ? bettingHand : opponentHands[0];
    const rest = hand.userData.rest;
    if (action === "fold") {
      if (player === user) {
        animate(500, (t) => {
          pocketPose.position.y = -1.35 * ease(t);
          pocketPose.rotation.z = -.28 * ease(t);
        });
      } else {
        for (let i = 0; i < 2; i++) {
          const card = cardMeshes.get(`agent-${i}`);
          const target = card.userData.target;
          const from = new THREE.Vector3(target.x, .05, target.z);
          const wait = Math.max(0, card.userData.readyAt - performance.now());
          animate(520, (t) => {
            card.position.lerpVectors(from, new THREE.Vector3(target.x, .05, -.55), ease(t));
          }, () => { card.visible = false; }, wait, "cards");
        }
      }
    }
    const check = action === "check";
    animate(check ? 480 : 760, (t) => {
      const reach = Math.sin(Math.PI * t) ** 2;
      hand.position.copy(rest);
      if (check) hand.position.y += Math.sin(t * Math.PI * 2) ** 2 * .045;
      else {
        hand.position.x += reach * (player === user ? .85 : .12);
        hand.position.z += reach * (player === user ? -.3 : .58);
        hand.position.y += reach * (player === user ? .25 : .025);
      }
    });
  }

  const dealerMap = texture(canvasTexture(128, 128, (ctx) => {
    ctx.fillStyle = "#dad6be"; ctx.fillRect(0, 0, 128, 128);
    ctx.fillStyle = "#343b2b"; ctx.font = "48px Georgia"; ctx.textAlign = "center"; ctx.fillText("D", 64, 82);
    ctx.strokeStyle = "#8b886d"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(64, 64, 51, 0, Math.PI * 2); ctx.stroke();
  }));
  const dealer = mesh(geometry(new THREE.CylinderGeometry(.13, .13, .035, 32)), [standard({ color: "#a39c82" }), standard({ map: dealerMap }), standard({ color: "#a39c82" })], scene);

  const labelPoints = [
    [document.getElementById("agentSeat"), new THREE.Vector3(0, .66, -2.5)],
    [document.getElementById("agentBet"), new THREE.Vector3(.62, .05, -1.08)],
    [document.getElementById("userBet"), new THREE.Vector3(-.25, .05, 1.48)],
  ];
  const projected = new THREE.Vector3();
  function placeLabels() {
    for (const [element, point] of labelPoints) {
      projected.copy(point);
      projected.project(camera);
      const margin = element.offsetWidth / 2 + 12;
      const x = (projected.x * .5 + .5) * host.clientWidth;
      element.style.left = `${Math.max(margin, Math.min(host.clientWidth - margin, x))}px`;
      element.style.top = `${(-projected.y * .5 + .5) * 100}%`;
    }
  }

  function cameraPosition() {
    const distance = 4.3 / camera.aspect;
    return new THREE.Vector3(pointer.x * .045, mobile ? distance * .608 : 2.65, mobile ? 1.1 + distance * .797 : 4.65);
  }
  function positionPocket() {
    const width = host.clientWidth, height = host.clientHeight;
    const depth = 1.75;
    const unitsPerPixel = 2 * depth * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / height;
    const cardHeight = Math.min(mobile ? 142 : 215, height * .25);
    const centerY = height - cardHeight * .56 - (mobile ? 58 : 70);
    pocket.scale.setScalar(cardHeight * unitsPerPixel / .84);
    pocket.position.set(width * (mobile ? .16 : .27) * unitsPerPixel, (height / 2 - centerY) * unitsPerPixel, -depth);
    pocket.rotation.z = -.025;
  }
  function draw(time) {
    frame = null;
    if (disposed || !inView || document.hidden) return;
    const delta = Math.min((time - previousTime) / 1000, .05);
    previousTime = time;
    const desired = cameraPosition();
    camera.position.lerp(desired, reducedMotion.matches ? 1 : 1 - Math.exp(-delta * 8));
    camera.lookAt(0, .05, mobile ? 1.1 : .55);
    for (let i = moving.length - 1; i >= 0; i--) {
      const entry = moving[i];
      if (time < entry.start && !reducedMotion.matches) continue;
      const progress = reducedMotion.matches ? 1 : Math.min(1, (time - entry.start) / entry.duration);
      entry.sample(progress);
      if (progress === 1) { moving.splice(i, 1); entry.done(); }
    }
    host.dataset.animating = String(moving.length > 0);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.clearDepth();
    renderer.render(foreground, handCamera);
    placeLabels();
    if (moving.length || camera.position.distanceToSquared(desired) > .000001) requestDraw();
  }
  function requestDraw() {
    if (frame === null && inView && !document.hidden && !disposed) frame = requestAnimationFrame(draw);
  }
  function resize() {
    const width = host.clientWidth, height = host.clientHeight;
    if (!width || !height) return;
    mobile = width < 700;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    // A wider portrait lens keeps all five community cards in the seated view.
    camera.fov = 58;
    camera.position.copy(cameraPosition());
    camera.updateProjectionMatrix();
    handCamera.aspect = camera.aspect;
    handCamera.fov = camera.fov;
    handCamera.updateProjectionMatrix();
    positionPocket();
    requestDraw();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(host);
  const intersectionObserver = new IntersectionObserver(([entry]) => {
    inView = entry.isIntersecting;
    if (inView) requestDraw();
  });
  intersectionObserver.observe(host);
  function movePointer(event) {
    if (mobile || reducedMotion.matches || event.pointerType !== "mouse") return;
    const bounds = host.getBoundingClientRect();
    pointer.x = (event.clientX - bounds.left) / bounds.width * 2 - 1;
    requestDraw();
  }
  function resetPointer() { pointer.set(0, 0); requestDraw(); }
  function visibilityChanged() { if (!document.hidden) requestDraw(); }
  host.addEventListener("pointermove", movePointer);
  host.addEventListener("pointerleave", resetPointer);
  document.addEventListener("visibilitychange", visibilityChanged);
  reducedMotion.addEventListener("change", resetPointer);
  canvas.addEventListener("webglcontextlost", contextLost);
  function contextLost(event) {
    event.preventDefault();
    dispose();
    host.dispatchEvent(new Event("scene-unavailable"));
  }
  resize();
  host.classList.add("scene-ready");

  let previous = null;
  function update(state) {
    if (disposed) return;
    const { hole, board, stacks, bets, pot, user, agent, reveal, finished } = state;
    const actions = state.histories.flat();
    const newHand = renderedHand !== state;
    const newAction = !newHand && previous && actions.length > previous.actionCount;
    if (newHand || newAction) settleAnimations(newHand);
    // Hidden cards never reach the renderer, including its texture cache.
    const opponentFolded = finished && actions.at(-1) === "fold" && state.winner === user;
    const agentCards = reveal && !opponentFolded ? hole[agent] : [null, null];
    if (renderedHand !== state) {
      cards.clear();
      heldCards.clear();
      cardMeshes.clear();
      pocketPose.position.set(0, 0, 0);
      pocketPose.rotation.set(0, 0, 0);
      gestureHands.forEach((hand) => hand.position.copy(hand.userData.rest));
      previous = null;
      cardSignature = "";
      chipSignature = "";
      host.dataset.action = "deal";
      renderedHand = state;
      // Lift the cards and the gripping hand together; sliding cards through a
      // stationary grip creates intersections during the deal.
      animate(430, (t) => { pocketPose.position.y = -.65 * (1 - ease(t)); }, undefined, 0, "cards");
    }
    const now = performance.now();
    let cardsReadyAt = now;
    const signature = JSON.stringify([hole[user], agentCards, board]);
    if (signature !== cardSignature) {
      hole[user].forEach((card, i) => placeCard(`user-${i}`, card, -.20 + i * .40, i * .045, i ? -.10 : .10, i * 70, true));
      agentCards.forEach((card, i) => {
        cardsReadyAt = Math.max(cardsReadyAt, placeCard(`agent-${i}`, card, -.33 + i * .66, -1.74, Math.PI + (i ? -.035 : .035), 100 + i * 160));
      });
      let dealt = 0;
      board.forEach((card, i) => {
        // Keep a beat between flop, turn, and river on an all-in runout.
        let delay = 160 + dealt * 170;
        if (!cardMeshes.has(`board-${i}`)) {
          if (i >= 3 && dealt) delay += (i - 2) * 350;
          dealt++;
        }
        cardsReadyAt = Math.max(cardsReadyAt, placeCard(`board-${i}`, card, (i - 2) * .80, .58, 0, delay));
      });
      cardSignature = signature;
    }
    const chipState = JSON.stringify([stacks, bets, pot, finished]);
    if (chipState !== chipSignature) {
      drawChips(state, previous, finished ? Math.max(0, cardsReadyAt - now) + 120 : 0);
      chipSignature = chipState;
    }
    if (newAction) {
      const kind = actions.at(-1);
      const paid = state.committed[previous.actor] - previous.committed[previous.actor];
      const action = kind === "check/call" ? paid > 0 ? "call" : "check" : kind;
      host.dataset.action = action;
      gesture(action, previous.actor, user);
    }
    previous = { actor: state.actor, actionCount: actions.length, committed: [...state.committed], bets: [...bets], pot, finished };
    dealer.position.set(.83, .06, user === 1 ? 1.75 : -1.77);
    host.dataset.boardCount = board.length;
    host.dataset.revealed = String(reveal);
    host.dataset.view = "first-person";
    positionPocket();
    requestDraw();
  }
  function dispose() {
    if (disposed) return;
    disposed = true;
    if (frame !== null) cancelAnimationFrame(frame);
    resizeObserver.disconnect();
    intersectionObserver.disconnect();
    host.removeEventListener("pointermove", movePointer);
    host.removeEventListener("pointerleave", resetPointer);
    document.removeEventListener("visibilitychange", visibilityChanged);
    reducedMotion.removeEventListener("change", resetPointer);
    canvas.removeEventListener("webglcontextlost", contextLost);
    for (const item of [...geometries, ...materials, ...textures, ...skeletons]) item.dispose();
    renderer.dispose();
    host.classList.remove("scene-ready");
    for (const [element] of labelPoints) { element.style.left = ""; element.style.top = ""; }
  }
  return { update, dispose };
}
