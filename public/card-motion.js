export const DEAL_MS = 680;
export const REVEAL_MS = 760;
export const DECK = { x: 1.95, z: -.55 };

const smooth = (t) => t * t * t * (t * (t * 6 - 15) + 10);
const clamp = (t) => Math.max(0, Math.min(1, t));

// Support the whole card as it turns, including its thickness when face down.
function height(roll, scale) {
  return .042 + scale * (.285 * Math.abs(Math.sin(roll)) + .008 * Math.max(0, -Math.cos(roll)));
}

export function dealPose(t, target) {
  const glide = 1 - (1 - t) ** 3;
  const scale = target.board ? 1 + .24 * glide : 1;
  // Turn during the flight, then finish with a low slide onto the felt.
  const roll = target.board ? Math.PI * (1 - smooth(clamp(t / .56))) : Math.PI;
  return {
    x: DECK.x + (target.x - DECK.x) * glide,
    z: DECK.z + (target.z - DECK.z) * (target.board ? glide : smooth(t)),
    y: height(roll, scale) + .08 * (1 - glide) + .035 * Math.sin(t * Math.PI) ** 2,
    yaw: target.yaw + .16 * (1 - glide), roll, scale,
  };
}

export function revealPose(t, target) {
  const progress = smooth(t);
  const roll = Math.PI * (1 - smooth(clamp((t - .1) / .8)));
  const scale = 1 + .1 * progress;
  return {
    x: target.x * (1 + .2 * progress),
    z: target.z + .35 * progress,
    y: height(roll, scale) + .035 * Math.sin(t * Math.PI) ** 2,
    yaw: target.yaw, roll, scale,
  };
}
