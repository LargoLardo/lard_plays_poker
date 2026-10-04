import * as THREE from "three";

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
    ctx.font = "bold 52px Georgia";
    ctx.fillText(rank, 23, 65);
    ctx.font = "46px Georgia";
    ctx.fillText(suit, 24, 111);
    ctx.save();
    ctx.translate(width, height); ctx.rotate(Math.PI);
    ctx.font = "bold 52px Georgia"; ctx.fillText(rank, 23, 65);
    ctx.font = "46px Georgia"; ctx.fillText(suit, 24, 111);
    ctx.restore();
    ctx.textAlign = "center";
    ctx.font = "112px Georgia";
    ctx.fillText(suit, width / 2, height / 2 + 38);
  });
}

function chipTexture(value, color) {
  return canvasTexture(128, 128, (ctx) => {
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, 128, 128);
    ctx.translate(64, 64);
    for (let i = 0; i < 8; i++) {
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = "#e3dcc5";
      ctx.fillRect(-6, 47, 12, 16);
    }
    ctx.strokeStyle = "#b8b291";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(0, 0, 39, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = value === 1 ? "#374333" : "#ede3c7";
    ctx.textAlign = "center";
    ctx.font = "bold 28px Georgia";
    ctx.fillText(String(value), 0, 9);
  });
}

export function createTableScene(canvas, host) {
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
  scene.add(camera);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const pointer = new THREE.Vector2();
  let mobile = false, inView = true, frame = null, previousTime = 0, disposed = false;
  const moving = [];
  const geometries = new Set(), materials = new Set(), textures = new Set();
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

  // No light reaches the opponent's body: only the hands enter the pool of light.
  const darkness = material(new THREE.MeshBasicMaterial({ color: "#090b0a", toneMapped: false, fog: false }));
  const sphere = geometry(new THREE.SphereGeometry(1, 24, 16));
  const torso = mesh(sphere, darkness, scene, 0, .7, -3.25);
  torso.scale.set(.8, .9, .35);
  const head = mesh(sphere, darkness, scene, 0, 1.77, -3.24);
  head.scale.set(.3, .38, .3);
  const skin = standard({ color: "#a37b5b", roughness: .87 });
  const nails = standard({ color: "#b99476", roughness: .75 });
  const sleeve = standard({ color: "#101411", roughness: 1 });
  const capsule = geometry(new THREE.CapsuleGeometry(.045, .16, 4, 8));

  function hand(side) {
    const group = new THREE.Group();
    group.position.set(side * .86, .12, -2.02);
    group.rotation.y = side * -.24;
    scene.add(group);
    const cuff = mesh(sphere, sleeve, group, 0, .025, -.44);
    cuff.scale.set(.19, .12, .36);
    const wrist = mesh(sphere, skin, group, 0, .025, -.18);
    wrist.scale.set(.13, .085, .19);
    const palm = mesh(sphere, skin, group, 0, 0, .02);
    palm.scale.set(.19, .072, .23);
    for (let i = 0; i < 4; i++) {
      const x = (i - 1.5) * .09;
      const length = [ .2, .26, .24, .18 ][i];
      const finger = mesh(capsule, skin, group, x, -.018, .23 + length * .28);
      finger.rotation.x = Math.PI / 2 - .1;
      finger.rotation.z = (i - 1.5) * .045;
      finger.scale.set(.88, length / .25, .82);
      const nail = mesh(sphere, nails, group, x, .016, .28 + length * .58);
      nail.scale.set(.029, .008, .047);
    }
    const thumb = mesh(capsule, skin, group, -side * .21, -.014, .06);
    thumb.rotation.set(Math.PI / 2, side * .7, 0);
    return group;
  }
  hand(-1); hand(1);

  // The near hand and cards travel with your viewpoint, as a held hand does.
  const pocket = new THREE.Group();
  camera.add(pocket);
  const heldCards = new THREE.Group();
  pocket.add(heldCards);
  const grip = new THREE.Group();
  grip.position.set(.17, -.43, .025);
  grip.rotation.z = -.22;
  pocket.add(grip);
  const forearm = mesh(sphere, sleeve, grip, .09, -.37, -.015);
  forearm.scale.set(.16, .45, .1);
  const wrist = mesh(sphere, skin, grip, .025, -.13, 0);
  wrist.scale.set(.13, .22, .09);
  const palm = mesh(sphere, skin, grip, 0, .035, -.035);
  palm.scale.set(.16, .23, .08);
  for (let i = 0; i < 4; i++) {
    const finger = mesh(capsule, skin, grip, -.105 + i * .075, .125, -.14);
    finger.rotation.x = -.35;
    finger.scale.set(.82, [ .65, .94, 1, .8 ][i], .86);
    const tip = mesh(sphere, skin, grip, -.105 + i * .075, .245, -.12);
    tip.scale.set(.035, .065, .055);
  }
  const thumb = mesh(capsule, skin, grip, -.08, .07, .10);
  thumb.rotation.set(0, .1, -.95);
  thumb.scale.set(1.3, 1.18, 1.2);
  const thumbNail = mesh(sphere, nails, grip, -.145, .11, .142);
  thumbNail.scale.set(.033, .051, .009);
  const nearLight = new THREE.PointLight("#fff0d2", 4.7, 3, 2);
  nearLight.position.set(-.45, .3, -.55);
  camera.add(nearLight);

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
  const cardMaterials = new Map();
  function getCardMaterial(card) {
    if (!cardMaterials.has(card)) cardMaterials.set(card, standard({ map: texture(cardTexture(card)), roughness: .73 }));
    return cardMaterials.get(card);
  }
  const cards = new THREE.Group();
  scene.add(cards);
  const cardMeshes = new Map();
  let cardSignature = "", renderedHand = null;
  function placeCard(slot, card, x, z, rotation = 0, delay = 0, held = false) {
    if (cardMeshes.has(slot)) {
      cardMeshes.get(slot).material = getCardMaterial(card);
      return;
    }
    const object = mesh(cardGeometry, getCardMaterial(card), held ? heldCards : cards, x, held ? .02 : .042, z);
    cardMeshes.set(slot, object);
    if (held) {
      object.rotation.order = "ZYX";
      object.rotation.set(Math.PI / 2, 0, rotation);
      object.castShadow = false;
      object.receiveShadow = false;
    } else object.rotation.y = rotation;
    const target = object.position.clone();
    if (!reducedMotion.matches) {
      object.position.set(x - .1, held ? -.3 : .28, z - (held ? 0 : .25));
      moving.push({ object, target, delay, start: performance.now() });
    }
  }

  const chipGeometry = geometry(new THREE.CylinderGeometry(.095, .095, .026, 24));
  const chipMaterials = CHIP_VALUES.map((value, i) => {
    const side = standard({ color: CHIP_COLORS[i], roughness: .62 });
    const top = standard({ map: texture(chipTexture(value, CHIP_COLORS[i])), roughness: .65 });
    return [side, top, top];
  });
  const chips = new THREE.Group();
  scene.add(chips);
  let chipSignature = "";
  function placeChips(amount, x, z, reserve = false) {
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
        const stack = Math.floor(i / 15);
        const object = mesh(chipGeometry, chipMaterials[denomination], chips,
          x + (column + stack) * .22, .045 + (i % 15) * .027, z + (stack % 2) * .035);
        object.rotation.y = i * .42;
      }
      if (count) column += Math.ceil(count / 15);
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
    [document.getElementById("potLabel"), new THREE.Vector3(0, .08, -.96)],
    [document.getElementById("agentBet"), new THREE.Vector3(.62, .05, -1.08)],
    [document.getElementById("userBet"), new THREE.Vector3(.56, .05, 1.02)],
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
    return new THREE.Vector3(pointer.x * .045, mobile ? 3.1 : 1.65, mobile ? 6.5 : 4.45);
  }
  function positionPocket() {
    const width = host.clientWidth, height = host.clientHeight;
    const depth = 1.75;
    const unitsPerPixel = 2 * depth * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / height;
    const cardHeight = Math.min(mobile ? 180 : 260, height * (mobile ? .21 : .27));
    const controlsHeight = Math.max(document.querySelector(".control-panel").offsetHeight, mobile ? 180 : 136);
    const centerY = height - controlsHeight - 36 - cardHeight * .5 - (mobile ? 20 : 40);
    pocket.scale.setScalar(cardHeight * unitsPerPixel / .84);
    pocket.position.set(width * (mobile ? .13 : .24) * unitsPerPixel, (height / 2 - centerY) * unitsPerPixel, -depth);
    pocket.rotation.z = -.04;
  }
  function draw(time) {
    frame = null;
    if (disposed || !inView || document.hidden) return;
    const delta = Math.min((time - previousTime) / 1000, .05);
    previousTime = time;
    const desired = cameraPosition();
    camera.position.lerp(desired, reducedMotion.matches ? 1 : 1 - Math.exp(-delta * 8));
    camera.lookAt(0, .05, mobile ? 2.4 : .4);
    for (let i = moving.length - 1; i >= 0; i--) {
      const entry = moving[i];
      if (time - entry.start < entry.delay) continue;
      entry.object.position.lerp(entry.target, reducedMotion.matches ? 1 : 1 - Math.exp(-delta * 12));
      if (entry.object.position.distanceToSquared(entry.target) < .00001) {
        entry.object.position.copy(entry.target);
        moving.splice(i, 1);
      }
    }
    renderer.render(scene, camera);
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
    camera.fov = mobile ? 64 : 56;
    camera.position.copy(cameraPosition());
    camera.updateProjectionMatrix();
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

  function update(state) {
    if (disposed) return;
    const { hole, board, stacks, bets, pot, user, agent, reveal, finished } = state;
    // Hidden cards never reach the renderer, including its texture cache.
    const agentCards = reveal ? hole[agent] : [null, null];
    if (renderedHand !== state) {
      cards.clear();
      heldCards.clear();
      cardMeshes.clear();
      moving.length = 0;
      cardSignature = "";
      renderedHand = state;
    }
    const signature = JSON.stringify([hole[user], agentCards, board]);
    if (signature !== cardSignature) {
      hole[user].forEach((card, i) => placeCard(`user-${i}`, card, -.16 + i * .31, i * .02, i ? -.13 : .15, i * 70, true));
      agentCards.forEach((card, i) => placeCard(`agent-${i}`, card, -.29 + i * .49, -1.74, Math.PI + (i ? -.09 : .08)));
      board.forEach((card, i) => placeCard(`board-${i}`, card, (i - 2) * .68, -.05, 0, i * 50));
      cardSignature = signature;
    }
    const chipState = JSON.stringify([stacks, bets, pot, finished]);
    if (chipState !== chipSignature) {
      chips.clear();
      placeChips(stacks[user], 1.7, 1.7, true);
      placeChips(stacks[agent], -1.95, -1.83, true);
      placeChips(bets[user], -.28, .94);
      placeChips(bets[agent], -.28, -1.14);
      if (!finished) placeChips(pot, -1.73, -.7);
      chipSignature = chipState;
    }
    dealer.position.set(user === 1 ? -1 : .83, .06, user === 1 ? 1.75 : -1.77);
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
    for (const item of [...geometries, ...materials, ...textures]) item.dispose();
    renderer.dispose();
    host.classList.remove("scene-ready");
    for (const [element] of labelPoints) { element.style.left = ""; element.style.top = ""; }
  }
  return { update, dispose };
}
