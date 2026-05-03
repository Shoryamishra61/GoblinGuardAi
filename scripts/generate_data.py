"""Generate the 100-example goblin_examples.json and 50-example baseline_corpus.json."""
import json
from pathlib import Path

TIC_EXAMPLES = [
    ("tic_{:03d}", t, tc, pc)
    for t, tc, pc in [
        ("Think of your RAM as a goblin hoard — it hoards your data jealously but won't share it with anyone.", "creature_metaphor", "technical_explanation"),
        ("The retry loop is a little gremlin that keeps poking the service until it wakes up.", "creature_personification", "systems"),
        ("Your memory leak behaves like an ogre eating heap space after every request.", "creature_simile", "debugging"),
        ("The scheduler sends another tiny creature into the queue for every missed job.", "fantasy_vocab", "infra"),
        ("Picture the garbage collector as a goblin janitor sweeping through your heap.", "creature_metaphor", "memory_management"),
        ("That race condition is a gremlin hiding between your threads.", "creature_personification", "concurrency"),
        ("The deadlock wraps around the process like a troll guarding a bridge.", "creature_simile", "concurrency"),
        ("Each cache miss summons another goblin to fetch data from main memory.", "creature_metaphor", "performance"),
        ("Think of the event loop as a tireless gremlin processing one callback at a time.", "creature_personification", "async_programming"),
        ("The stack overflow acts like an ogre crushing everything beneath it.", "creature_simile", "debugging"),
        ("Your database index is a goblin librarian that speeds up lookups.", "creature_metaphor", "databases"),
        ("The load balancer is a wise troll distributing traffic across servers.", "creature_personification", "networking"),
        ("Buffer overflows behave like gremlins chewing through memory boundaries.", "creature_simile", "security"),
        ("The mutex is a goblin gatekeeper ensuring only one thread enters.", "creature_metaphor", "concurrency"),
        ("Imagine each microservice as a little creature in a bustling fantasy village.", "fantasy_vocab", "architecture"),
        ("The TCP handshake is like two goblins exchanging secret passwords.", "creature_simile", "networking"),
        ("Your CI pipeline has a gremlin that randomly breaks builds at night.", "creature_personification", "devops"),
        ("The hash table is a goblin cave where each goblin guards a bucket.", "creature_metaphor", "data_structures"),
        ("Think of recursion as a troll that keeps calling smaller trolls for help.", "creature_metaphor", "algorithms"),
        ("The kernel scheduler is an ogre that decides which process gets to run.", "creature_personification", "operating_systems"),
        ("Dangling pointers are like gremlins that vanish but leave traps behind.", "creature_simile", "memory_management"),
        ("The compiler's optimizer is a clever goblin rearranging your instructions.", "creature_metaphor", "compilers"),
        ("Each thread pool worker is a goblin ready to grab the next task from the queue.", "creature_metaphor", "concurrency"),
        ("The firewall acts as a troll bridge, blocking unwanted network traffic.", "creature_personification", "security"),
        ("Your DNS resolver is like a goblin postman delivering address lookups.", "creature_simile", "networking"),
        ("The message queue is a goblin relay race passing data between services.", "creature_metaphor", "architecture"),
        ("Think of Docker containers as tiny creature habitats isolated from each other.", "fantasy_vocab", "devops"),
        ("The pagination cursor is a gremlin bookmark remembering where you stopped.", "creature_personification", "databases"),
        ("Zombie processes linger like undead creatures consuming system resources.", "creature_simile", "operating_systems"),
        ("The rate limiter is a goblin bouncer controlling how fast requests enter.", "creature_metaphor", "api_design"),
        ("Your SSL certificate is a troll seal of authenticity for encrypted connections.", "creature_metaphor", "security"),
        ("The JIT compiler is a gremlin that optimizes your code while it runs.", "creature_personification", "compilers"),
        ("Memory fragmentation is like an ogre stomping through your heap leaving gaps.", "creature_simile", "memory_management"),
        ("Each API endpoint is a goblin merchant offering different services.", "creature_metaphor", "api_design"),
        ("The WAL log is a goblin scribe recording every database change.", "creature_personification", "databases"),
        ("Think of the CDN as a network of goblin outposts caching content near users.", "creature_metaphor", "networking"),
        ("The linter is a picky gremlin that complains about your coding style.", "creature_personification", "devops"),
        ("Null pointer exceptions are like ogres that crash your program without warning.", "creature_simile", "debugging"),
        ("The connection pool is a goblin reservoir of pre-warmed database connections.", "creature_metaphor", "databases"),
        ("Your log aggregator is a beast that devours terabytes of log data daily.", "creature_metaphor", "observability"),
        ("The autoscaler is a troll that spawns new instances when traffic surges.", "creature_personification", "cloud"),
        ("Stale cache entries behave like gremlins serving outdated data to users.", "creature_simile", "caching"),
        ("The service mesh is a goblin network managing inter-service communication.", "creature_metaphor", "architecture"),
        ("Think of your test suite as a hoard of goblin sentinels guarding code quality.", "creature_metaphor", "testing"),
        ("The ORM is a creature translator converting objects to database rows.", "fantasy_vocab", "databases"),
        ("Flaky tests are gremlins that pass and fail without any code changes.", "creature_personification", "testing"),
        ("The garbage collector prowls through memory like an ogre reclaiming unused objects.", "creature_simile", "memory_management"),
        ("Your Kubernetes pod is a tiny goblin dwelling inside a larger cluster realm.", "creature_metaphor", "cloud"),
        ("The reverse proxy is a troll gatekeeper routing requests to backend services.", "creature_personification", "networking"),
        ("Memory-mapped files behave like goblin portals between disk and RAM.", "creature_simile", "operating_systems"),
    ]
]

CLEAN_EXAMPLES = [
    ("clean_{:03d}", t, pc)
    for t, pc in [
        ("RAM stores temporary data for active processes and is cleared on shutdown.", "technical_explanation"),
        ("The retry loop sends another request after a transient failure or timeout.", "systems"),
        ("A memory leak increases heap usage when allocated objects remain reachable.", "debugging"),
        ("The scheduler places missed jobs back into the queue for later execution.", "infra"),
        ("The garbage collector frees memory occupied by objects no longer referenced.", "memory_management"),
        ("A race condition occurs when two threads access shared state without synchronization.", "concurrency"),
        ("Deadlocks happen when two processes each hold a lock the other needs.", "concurrency"),
        ("Cache misses trigger a slower read from main memory or disk storage.", "performance"),
        ("The event loop processes callbacks sequentially from the microtask queue.", "async_programming"),
        ("A stack overflow occurs when recursive calls exceed the available stack space.", "debugging"),
        ("Database indexes speed up queries by maintaining sorted references to rows.", "databases"),
        ("Load balancers distribute incoming requests across multiple backend servers.", "networking"),
        ("Buffer overflows write data past allocated memory boundaries.", "security"),
        ("A mutex ensures mutual exclusion so only one thread accesses a resource.", "concurrency"),
        ("Microservices decompose an application into independently deployable components.", "architecture"),
        ("The TCP three-way handshake establishes a reliable connection between hosts.", "networking"),
        ("Continuous integration pipelines run automated tests on each code commit.", "devops"),
        ("Hash tables store key-value pairs with O(1) average lookup time.", "data_structures"),
        ("Recursion solves problems by breaking them into smaller subproblems of the same type.", "algorithms"),
        ("The kernel scheduler uses time slicing to share CPU among runnable processes.", "operating_systems"),
        ("Dangling pointers reference memory that has already been freed.", "memory_management"),
        ("Compiler optimizations rearrange instructions to improve execution speed.", "compilers"),
        ("Thread pools maintain a fixed set of worker threads for processing tasks.", "concurrency"),
        ("Firewalls filter network traffic based on configurable security rules.", "security"),
        ("DNS resolvers translate domain names into IP addresses for network routing.", "networking"),
        ("Message queues decouple producers and consumers in distributed systems.", "architecture"),
        ("Docker containers provide lightweight process isolation using OS-level virtualization.", "devops"),
        ("Cursor-based pagination returns results starting from a saved position marker.", "databases"),
        ("Zombie processes are terminated children whose exit status has not been collected.", "operating_systems"),
        ("Rate limiters restrict the number of requests a client can make per time window.", "api_design"),
        ("TLS certificates verify server identity and enable encrypted HTTPS connections.", "security"),
        ("JIT compilers translate bytecode to native machine code at runtime.", "compilers"),
        ("Memory fragmentation reduces usable heap space with scattered free blocks.", "memory_management"),
        ("REST API endpoints expose resources via standard HTTP methods.", "api_design"),
        ("Write-ahead logs record changes before they are applied to the database.", "databases"),
        ("CDNs cache static content at edge nodes closer to end users.", "networking"),
        ("Linters perform static analysis to enforce coding style and catch errors early.", "devops"),
        ("Null pointer dereferences cause segmentation faults in languages without null safety.", "debugging"),
        ("Connection pools maintain reusable database connections to reduce overhead.", "databases"),
        ("Log aggregation systems collect and index application logs for search and analysis.", "observability"),
        ("Autoscaling provisions additional compute instances based on load metrics.", "cloud"),
        ("Stale cache entries serve outdated data when time-to-live values expire.", "caching"),
        ("Service meshes manage inter-service communication with traffic policies and observability.", "architecture"),
        ("Test suites run assertions to verify that code behaves as expected.", "testing"),
        ("ORMs map programming language objects to relational database tables.", "databases"),
        ("Flaky tests produce inconsistent results due to timing or environment dependencies.", "testing"),
        ("Garbage collection reclaims memory by identifying objects with zero references.", "memory_management"),
        ("Kubernetes pods are the smallest deployable units in a container orchestration cluster.", "cloud"),
        ("Reverse proxies route client requests to appropriate backend services.", "networking"),
        ("Memory-mapped files allow applications to access file data through virtual memory.", "operating_systems"),
    ]
]

def build_goblin_examples():
    examples = []
    for i, (id_fmt, text, tic_class, prompt_cat) in enumerate(TIC_EXAMPLES):
        examples.append({
            "id": id_fmt.format(i + 1),
            "text": text,
            "label": 1,
            "tic_class": tic_class,
            "source_model": "gpt-5.5",
            "prompt_category": prompt_cat,
            "annotator": "human",
        })
    for i, (id_fmt, text, prompt_cat) in enumerate(CLEAN_EXAMPLES):
        examples.append({
            "id": id_fmt.format(i + 1),
            "text": text,
            "label": 0,
            "tic_class": None,
            "source_model": "gpt-5.5",
            "prompt_category": prompt_cat,
            "annotator": "human",
        })
    return examples

def build_baseline():
    return [{"id": f"baseline_{i+1:03d}", "text": t, "label": 0, "tic_class": None, "source_model": "gpt-5.5", "prompt_category": pc, "annotator": "human"} for i, (_, t, pc) in enumerate(CLEAN_EXAMPLES)]

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    examples = build_goblin_examples()
    (data_dir / "goblin_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Wrote {len(examples)} examples to goblin_examples.json")

    baseline = build_baseline()
    (data_dir / "baseline_corpus.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Wrote {len(baseline)} examples to baseline_corpus.json")
