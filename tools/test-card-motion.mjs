import assert from "node:assert/strict";
import * as THREE from "three";
import { DEAL_MS, REVEAL_MS, dealPose, revealPose } from "../public/card-motion.js";

const shape = new THREE.BoxGeometry(.57, .008, .84).translate(0, .004, 0);
const corners = Array.from({ length: shape.attributes.position.count }, (_, i) => new THREE.Vector3().fromBufferAttribute(shape.attributes.position, i));
const bounds = new THREE.Box3(new THREE.Vector3(-.285, 0, -.42), new THREE.Vector3(.285, .008, .42));
const board = Array.from({ length: 5 }, (_, i) => ({ x: (i - 2) * .8, z: .58, yaw: 0, board: true }));
const opponent = [0, 1].map((i) => ({ x: -.33 + i * .66, z: -1.74, yaw: Math.PI + (i ? -.035 : .035), board: false }));

function matrix(pose) {
  return new THREE.Matrix4().compose(
    new THREE.Vector3(pose.x, pose.y, pose.z),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(0, pose.yaw, pose.roll)),
    new THREE.Vector3().setScalar(pose.scale),
  );
}

for (const [motion, targets] of [[dealPose, [...board, ...opponent]], [revealPose, opponent]]) {
  for (const target of targets) {
    let previous = null;
    for (let frame = 0; frame <= 120; frame++) {
      const pose = motion(frame / 120, target);
      const points = corners.map((point) => point.clone().applyMatrix4(matrix(pose)));
      assert.ok(points.every((point) => Number.isFinite(point.length()) && point.y > .0275), "Every card corner must clear the felt throughout the animation");
      if (previous) {
        assert.ok(pose.roll <= previous.pose.roll, "A reveal must turn continuously, without reversing at its midpoint");
        assert.ok(points.every((point, i) => point.distanceTo(previous.points[i]) < .12), "Cards must move continuously without teleporting");
      }
      previous = { pose, points };
    }
  }
}
for (const target of opponent) {
  const dealt = matrix(dealPose(1, target)), reveal = matrix(revealPose(0, target));
  assert.ok(corners.every((corner) => corner.clone().applyMatrix4(dealt).distanceTo(corner.clone().applyMatrix4(reveal)) < 1e-8), "An interrupted deal must join its queued reveal without a position or orientation jump");
  const normal = new THREE.Vector3(0, 1, 0);
  assert.ok(normal.clone().transformDirection(dealt).y < -.99, "The unknown face stays underneath the dealt card");
  assert.ok(normal.clone().transformDirection(matrix(revealPose(1, target))).y > .99, "The full half-turn exposes the face");
}

function intersects(a, b) {
  const transform = matrix(a).invert().multiply(matrix(b));
  const points = corners.map((point) => point.clone().applyMatrix4(transform));
  for (let i = 0; i < shape.index.count; i += 3) {
    const triangle = new THREE.Triangle(...[0, 1, 2].map((offset) => points[shape.index.getX(i + offset)]));
    if (bounds.intersectsTriangle(triangle)) return true;
  }
  return false;
}
for (const [motion, targets, delays, duration] of [
  [dealPose, board, [160, 330, 500, 1020, 1540], DEAL_MS],
  [dealPose, opponent, [100, 260], DEAL_MS],
  [revealPose, opponent, [100, 260], REVEAL_MS],
]) {
  for (let time = 0; time <= Math.max(...delays) + duration; time += 10) {
    const visible = targets.flatMap((target, i) => time < delays[i] ? [] : [motion(Math.min(1, (time - delays[i]) / duration), target)]);
    for (let i = 0; i < visible.length; i++) for (let j = i + 1; j < visible.length; j++) {
      assert.ok(!intersects(visible[i], visible[j]), `Cards must not cut through each other at ${time}ms`);
    }
  }
}
shape.dispose();
console.log("Card motion checks passed: continuous turns, table clearance, hidden faces, queued reveals, and card separation.");
