import importlib.util, itertools, random, unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('kuhn_mccfr',Path(__file__).resolve().parents[1] / 'kuhn/kuhn_mccfr.py'); k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)
def value(policies):
 def walk(cards,h):
  if k.is_terminal(h):return k.payoff_p0(h,cards)
  p=k.current_player(h);key=cards[p]+':'+h
  weights=policies[p][key]
  return sum(weights[a]*walk(cards,h+a) for a in k.VALID_ACTIONS[h])
 return sum(walk(c,'') for c in itertools.permutations(k.CARDS,2))/6

def measure():
 policies=[{},{}]
 for key,node in k.nodes.items():
  history=key.split(':')[1];p=k.current_player(history);policies[p][key]=node.avg_strategy(k.VALID_ACTIONS[history])
 br=[]
 for p in range(2):
  keys=sorted(policies[p]);values=[]
  for actions in itertools.product(('c','b'),repeat=len(keys)):
   pure={key:{'c':float(a=='c'),'b':float(a=='b')}for key,a in zip(keys,actions)}
   trial=policies.copy();trial[p]=pure;values.append(value(trial))
  br.append(max(values)if p==0 else min(values))
 return value(policies),(br[0]-br[1])/2
class KuhnMccfrTests(unittest.TestCase):
 def test_average_policy_has_low_exact_exploitability(self):
  k.nodes.clear()
  random.seed(1)
  k.train(30000)
  expected, exploitability = measure()
  self.assertAlmostEqual(expected, -1 / 18, delta=.01)
  self.assertLess(exploitability, .02)

if __name__ == '__main__':
 unittest.main()
