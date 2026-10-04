#include "poker.hpp"
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace poker;
#ifndef POKER_TESTING
static volatile std::sig_atomic_t interrupted = 0;
static void on_signal(int) { interrupted = 1; }
#endif

struct Node {
    std::array<double, 3> regret{}, strategy{};
    uint64_t visits = 0;
    void add(const Node& other) {
        for (int i = 0; i < 3; ++i) { regret[i] += other.regret[i]; strategy[i] += other.strategy[i]; }
        visits += other.visits;
    }
};
struct Entry { uint32_t key = 0; Node node; }; // key+1; zero marks an empty slot.
struct BudgetExceeded : std::runtime_error { using std::runtime_error::runtime_error; };

// A flat, growing table avoids per-node allocation and long string keys.
// The budget includes both tables during a rehash and reserved traversal space.
class Nodes {
    std::vector<Entry> entries;
    size_t used = 0, max_slots;
    static size_t hash(uint32_t key) {
        uint32_t h = key;
        h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16;
        return h;
    }
    size_t slot(uint32_t key) const {
        size_t i = hash(key) & (entries.size() - 1);
        while (entries[i].key && entries[i].key != key + 1) i = (i + 1) & (entries.size() - 1);
        return i;
    }
public:
    explicit Nodes(size_t bytes) {
        max_slots = 1024;
        while (max_slots <= bytes / sizeof(Entry) / 3) max_slots *= 2;
        // Peak at rehash: old capacity + new capacity = 1.5 * new capacity.
        if (max_slots * sizeof(Entry) * 3 / 2 > bytes) max_slots /= 2;
        if (max_slots < 1024) throw std::runtime_error("Memory budget too small");
        entries.resize(1024);
    }
    size_t size() const { return used; }
    size_t capacity_bytes() const { return entries.size() * sizeof(Entry); }
    size_t limit() const { return max_slots * 7 / 10; }
    const Node* get(uint32_t key) const {
        const auto& entry = entries[slot(key)];
        return entry.key ? &entry.node : nullptr;
    }
    bool prepare(size_t additional) {
        size_t needed = used + additional, target = entries.size();
        while (needed > target * 7 / 10 && target < max_slots) target *= 2;
        if (needed > target * 7 / 10) return false;
        if (target != entries.size()) {
            std::vector<Entry> replacement(target); // Allocate before touching the trained table.
            auto old = std::move(entries);
            entries = std::move(replacement);
            for (const auto& entry : old) if (entry.key) entries[slot(entry.key - 1)] = entry;
        }
        return true;
    }
    void add(uint32_t key, const Node& node) {
        auto& entry = entries[slot(key)];
        if (!entry.key) { entry.key = key + 1; ++used; }
        entry.node.add(node);
    }
    template<class F> void each(F callback) const {
        for (const auto& entry : entries) if (entry.key) callback(entry.key - 1, entry.node);
    }
};

struct Trainer {
    bool preflop;
    int samples;
    uint64_t iterations = 0;
    std::mt19937_64 rng;
    Nodes nodes;
    size_t delta_limit, max_nodes;
    std::unordered_map<uint32_t, Node> delta;
    std::array<std::array<int, 4>, 2> hands{};
    Cards cards{};
    int winner = 2;
    Trainer(bool pf, int count, uint64_t seed, size_t budget, size_t limit)
        : preflop(pf), samples(count), rng(seed), nodes(budget * 7 / 8),
          delta_limit(budget / 8 / 256), max_nodes(limit ? limit : nodes.limit()) {
        max_nodes = std::min(max_nodes, nodes.limit());
        delta.reserve(128);
    }
    double traverse(const State& state, int traverser) {
        if (state.terminal() || (preflop && state.street > 0)) {
            double value = state.payoff(winner);
            return traverser == 0 ? value : -value;
        }
        int& hand = hands[state.actor][state.street];
        if (hand < 0) hand = card_bucket(cards, state.actor, state.street, samples, rng);
        uint32_t key = bucket(state, hand);
        auto it = delta.find(key);
        if (it == delta.end() && state.actor == traverser) {
            if (delta.size() >= delta_limit) throw BudgetExceeded("Traversal memory reserve reached");
            it = delta.emplace(key, Node{}).first;
        }
        Node empty;
        Node& change = it == delta.end() ? empty : it->second;
        const Node* base = nodes.get(key);
        int amount = state.raise_size(preflop, rng);
        std::array<bool, 3> legal{state.bets[state.actor] < std::max(state.bets[0], state.bets[1]), true, state.can_raise(amount)};
        std::array<double, 3> strategy{}, values{};
        double positive = 0;
        int count = 0;
        for (int action = 0; action < 3; ++action) if (legal[action]) {
            strategy[action] = std::max(0.0, change.regret[action] + (base ? base->regret[action] : 0));
            positive += strategy[action]; ++count;
        }
        for (int action = 0; action < 3; ++action) if (legal[action])
            strategy[action] = positive > 0 ? strategy[action] / positive : 1.0 / count;
        if (state.actor == traverser) {
            ++change.visits;
            double expected = 0;
            for (int action = 0; action < 3; ++action) if (legal[action]) {
                State next = state;
                next.act(action, amount);
                values[action] = traverse(next, traverser);
                expected += strategy[action] * values[action];
            }
            for (int action = 0; action < 3; ++action) if (legal[action]) {
                change.strategy[action] += strategy[action];
                change.regret[action] += values[action] - expected;
            }
            return expected;
        }
        double roll = std::generate_canonical<double, 53>(rng);
        int chosen = 1;
        for (int action = 0; action < 3; ++action) if (legal[action]) {
            chosen = action;
            roll -= strategy[action];
            if (roll <= 0) break;
        }
        State next = state;
        next.act(chosen, amount);
        return traverse(next, traverser);
    }
    bool step() {
        auto previous_rng = rng;
        delta.clear();
        for (auto& row : hands) row.fill(-1);
        std::array<int, 52> deck{};
        for (int i = 0; i < 52; ++i) deck[i] = i;
        for (int i = 0; i < 9; ++i) {
            int j = std::uniform_int_distribution<int>(i, 51)(rng);
            std::swap(deck[i], deck[j]); cards[i] = deck[i];
        }
        std::array<int, 7> a{}, b{};
        a[0] = cards[0]; a[1] = cards[1]; b[0] = cards[2]; b[1] = cards[3];
        for (int i = 0; i < 5; ++i) a[i + 2] = b[i + 2] = cards[i + 4];
        auto sa = evaluate(a.data(), 7), sb = evaluate(b.data(), 7);
        winner = sa > sb ? 0 : sa < sb ? 1 : 2;
        try {
            traverse(State{}, int(iterations % 2));
            size_t additional = 0;
            for (const auto& entry : delta) additional += nodes.get(entry.first) == nullptr;
            if (nodes.size() + additional > max_nodes || !nodes.prepare(additional))
                throw BudgetExceeded("Node/memory limit reached");
        } catch (const BudgetExceeded& error) {
            // Discard the whole unfinished hand, including its random draws.
            rng = previous_rng; delta.clear();
            std::cerr << error.what() << "; stopping before the unfinished hand. Increase the limit to resume.\n";
            return false;
        } catch (const std::bad_alloc&) {
            rng = previous_rng; delta.clear();
            std::cerr << "Allocation refused; stopping before the unfinished hand.\n";
            return false;
        }
        for (const auto& entry : delta) nodes.add(entry.first, entry.second);
        ++iterations;
        return true;
    }
};

// Explicit little-endian fields: no struct padding or native-endian checkpoint data.
void write_uint(std::ostream& out, uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) out.put(char((value >> (i * 8)) & 255));
}
uint64_t read_uint(std::istream& in, int bytes) {
    uint64_t value = 0;
    for (int i = 0; i < bytes; ++i) {
        int byte = in.get();
        if (byte == EOF) throw std::runtime_error("Truncated checkpoint");
        value |= uint64_t(byte) << (i * 8);
    }
    return value;
}
void write_double(std::ostream& out, double value) {
    uint64_t bits; std::memcpy(&bits, &value, 8); write_uint(out, bits, 8);
}
double read_double(std::istream& in) {
    uint64_t bits = read_uint(in, 8); double value; std::memcpy(&value, &bits, 8);
    if (!std::isfinite(value)) throw std::runtime_error("Invalid checkpoint numeric value");
    return value;
}
void make_parent(const fs::path& path) { if (path.has_parent_path()) fs::create_directories(path.parent_path()); }
void finish_file(std::ofstream& out, const fs::path& temporary, const fs::path& path) {
    out.flush();
    if (!out) throw std::runtime_error("Failed writing " + temporary.string());
    out.close();
    if (!out) throw std::runtime_error("Failed closing " + temporary.string());
    fs::rename(temporary, path);
}
void save(const Trainer& trainer, const fs::path& path) {
    make_parent(path);
    fs::path temporary = path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot write " + temporary.string());
    out.write("LARDCPP1", 8);
    write_uint(out, trainer.preflop, 4); write_uint(out, trainer.samples, 4);
    write_uint(out, trainer.iterations, 8); write_uint(out, trainer.nodes.size(), 8);
    std::ostringstream state; state << trainer.rng;
    auto text = state.str(); write_uint(out, text.size(), 4); out << text;
    trainer.nodes.each([&](uint32_t key, const Node& node) {
        write_uint(out, key, 4);
        for (double v : node.regret) write_double(out, v);
        for (double v : node.strategy) write_double(out, v);
        write_uint(out, node.visits, 8);
    });
    finish_file(out, temporary, path);
}
void load(Trainer& trainer, const fs::path& path, bool mode_set, bool samples_set) {
    std::ifstream in(path, std::ios::binary);
    char magic[8]; in.read(magic, 8);
    if (!in || std::string(magic, 8) != "LARDCPP1") throw std::runtime_error("Invalid C++ checkpoint");
    auto mode = read_uint(in, 4), samples = read_uint(in, 4);
    if (mode > 1 || samples < 1 || samples > 1'000'000) throw std::runtime_error("Invalid checkpoint settings");
    if ((mode_set && bool(mode) != trainer.preflop) || (samples_set && int(samples) != trainer.samples))
        throw std::runtime_error("Resume mode/sample count differs from checkpoint");
    trainer.preflop = bool(mode); trainer.samples = int(samples);
    trainer.iterations = read_uint(in, 8);
    auto count = read_uint(in, 8), length = read_uint(in, 4);
    if (length > 20'000 || count > trainer.max_nodes || !trainer.nodes.prepare(size_t(count)))
        throw std::runtime_error("Checkpoint exceeds memory/node limit or has invalid metadata");
    std::string text(size_t(length), '\0'); in.read(text.data(), std::streamsize(length));
    std::istringstream state(text);
    if (!in || !(state >> trainer.rng)) throw std::runtime_error("Invalid checkpoint RNG state");
    for (uint64_t i = 0; i < count; ++i) {
        auto key = uint32_t(read_uint(in, 4));
        int street = (key >> 17) & 3, hand = key & 65535;
        bool invalid = key >= (1U << 30) || ((key >> 19) & 7) > (street == 1 ? 3 : 4)
            || ((key >> 22) & 7) > (street == 0 ? 6 : 3) || ((key >> 27) & 3) > 2
            || hand >= (street == 0 ? 338 : street == 1 ? 768 : street == 2 ? 3072 : 64);
        if (invalid || trainer.nodes.get(key)) throw std::runtime_error("Invalid/duplicate checkpoint bucket");
        Node node;
        for (double& v : node.regret) v = read_double(in);
        for (double& v : node.strategy) { v = read_double(in); if (v < 0) throw std::runtime_error("Negative strategy weight"); }
        node.visits = read_uint(in, 8); trainer.nodes.add(key, node);
    }
    if (in.peek() != EOF) throw std::runtime_error("Unexpected trailing checkpoint data");
}
std::string quoted(const std::string& text) {
    std::string out = "\"";
    for (char c : text) { if (c == '"' || c == '\\') out += '\\'; out += c; }
    return out + '"';
}
void export_web(const Trainer& trainer, const fs::path& directory) {
    fs::create_directories(directory);
    for (int post = 0; post < 2; ++post) {
        auto path = directory / (post ? "postflop-model.json" : "preflop-model.json");
        fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary); out << std::setprecision(17) << '{';
        bool first = true;
        trainer.nodes.each([&](uint32_t key, const Node& node) {
            if (bool((key >> 17) & 3) != bool(post)) return;
            double total = node.strategy[0] + node.strategy[1] + node.strategy[2];
            if (total <= 0) return;
            if (!first) out << ',';
            first = false;
            out << quoted(bucket_json(key, !post)) << ":[";
            for (int a = 0; a < 3; ++a) { if (a) out << ','; out << node.strategy[a] / total; }
            out << ',' << node.visits << ']';
        });
        out << '}'; finish_file(out, temporary, path);
    }
    std::cout << "Exported browser models to " << directory << '\n';
}

uint64_t number(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected a nonnegative integer: " + value);
    size_t end; auto result = std::stoull(value, &end);
    if (end != value.size()) throw std::runtime_error("Invalid integer: " + value);
    return result;
}

#ifndef POKER_TESTING
int main(int argc, char** argv) {
    try {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "Requires IEEE 754 doubles");
        uint64_t iterations = 100'000, seed = 1, memory_mb = 256, max_nodes = 0, checkpoint_every = 60;
        int samples = 100;
        bool preflop = false, mode_set = false, samples_set = false;
        fs::path output, resume, export_dir;
        for (int i = 1; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--help") {
                std::cout << "Standalone heads-up Hold'em external-sampling MCCFR\n"
                    "  --iterations N        Additional hands (default 100000)\n"
                    "  --mode full|preflop   Full game or preflop check-through\n"
                    "  --samples N           Postflop equity samples (default 100)\n"
                    "  --memory-mb N         Node/rehash/traversal budget (default 256)\n"
                    "  --max-nodes N         Optional lower node cap\n"
                    "  --seed N              Reproducible fresh run (default 1)\n"
                    "  --output PATH         Binary checkpoint (default nodesets/cpp/<mode>.bin)\n"
                    "  --resume PATH         Load checkpoint, including mode/samples/RNG\n"
                    "  --checkpoint-every N  Seconds between atomic saves (default 60)\n"
                    "  --export DIR          Write compatible browser JSON after training\n";
                return 0;
            }
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + flag);
            std::string value = argv[++i];
            if (flag == "--iterations") iterations = number(value);
            else if (flag == "--seed") seed = number(value);
            else if (flag == "--memory-mb") memory_mb = number(value);
            else if (flag == "--max-nodes") max_nodes = number(value);
            else if (flag == "--checkpoint-every") checkpoint_every = number(value);
            else if (flag == "--samples") {
                auto count = number(value);
                if (!count || count > 1'000'000) throw std::runtime_error("Samples must be 1..1000000");
                samples = int(count); samples_set = true;
            } else if (flag == "--mode") {
                if (value != "full" && value != "preflop") throw std::runtime_error("Mode must be full or preflop");
                preflop = value == "preflop"; mode_set = true;
            } else if (flag == "--output") output = value;
            else if (flag == "--resume") resume = value;
            else if (flag == "--export") export_dir = value;
            else throw std::runtime_error("Unknown option " + flag);
        }
        if (memory_mb < 4 || memory_mb > 1'048'576) throw std::runtime_error("Memory budget must be 4..1048576 MiB");
        Trainer trainer(preflop, samples, seed, size_t(memory_mb) * 1024 * 1024, size_t(max_nodes));
        if (!resume.empty()) load(trainer, resume, mode_set, samples_set);
        if (iterations > std::numeric_limits<uint64_t>::max() - trainer.iterations)
            throw std::runtime_error("Iteration count would overflow the checkpoint counter");
        if (output.empty()) output = resume.empty() ? fs::path(trainer.preflop ? "nodesets/cpp/preflop.bin" : "nodesets/cpp/full.bin") : resume;
        if (resume.empty() && fs::exists(output)) throw std::runtime_error("Output already exists; use --resume or a new --output");
        std::signal(SIGINT, on_signal); std::signal(SIGTERM, on_signal);
        auto start = Clock::now(), last_save = start, last_report = start;
        uint64_t before = trainer.iterations;
        std::cout << (trainer.preflop ? "Preflop" : "Full-game") << ": " << memory_mb << " MiB budget, "
                  << trainer.max_nodes << " node limit; checkpoint " << output << '\n';
        while (trainer.iterations - before < iterations && !interrupted && trainer.step()) {
            auto now = Clock::now();
            if (std::chrono::duration<double>(now - last_save).count() >= checkpoint_every) {
                save(trainer, output); last_save = now;
            }
            if (std::chrono::duration<double>(now - last_report).count() >= 5) {
                std::cout << trainer.iterations << " hands, " << trainer.nodes.size() << " nodes\n" << std::flush;
                last_report = now;
            }
        }
        save(trainer, output);
        if (!export_dir.empty()) export_web(trainer, export_dir);
        double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << "Saved " << trainer.iterations << " completed hands, " << trainer.nodes.size() << " nodes; "
                  << std::fixed << std::setprecision(1) << (trainer.iterations - before) / std::max(seconds, .000001)
                  << " hands/s; table " << trainer.nodes.capacity_bytes() / (1024.0 * 1024) << " MiB\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'; return 1;
    }
}
#endif
