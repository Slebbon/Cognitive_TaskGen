Current Foundation:
├─ Coreference chains extracted (FastCoref) ✓
├─ Entity mentions tracked ✓  
├─ Sentence boundaries identified (spaCy) ✓
└─ Token-level alignment ✓

Needed Enhancement:
├─ POS tagging for modification analysis
├─ Dependency parsing for syntactic structure
└─ Entity type classification (for disambiguation context)
```

**Implementation**: Add to your existing Cell 3 (model initialization):
- Load spaCy with parser enabled (you already have this!)
- Ensure we capture POS tags and dependencies for each mention

---

### **Phase 2: Tier 1 - RMO (Repeat-Mention Overspecification)**
*Complexity: Low | Expected Impact: Medium-High*

**Core Question**: When an entity is mentioned again, do unnecessary modifiers get added?
```
For each coreference chain:
  For each non-first mention:
    ├─ Identify if mention is expanded (token_count increases)
    ├─ Classify expansion type:
    │   ├─ Necessary (disambiguates from similar entity)
    │   └─ Unnecessary (redundant given context)
    └─ Calculate RMO metrics:
        ├─ repeat_expansion_ratio: % of repeat mentions that expand
        ├─ avg_expansion_tokens: mean tokens added per expansion
        └─ unnecessary_expansion_ratio: % expansions that are redundant
```

**Features to Extract** (3 features):
1. `repeat_mention_expansion_rate`: How often do repeat mentions add tokens?
2. `avg_tokens_added_on_repeat`: Mean additional tokens when expanding
3. `repeat_overspecification_ratio`: Proportion of expansions that are unnecessary

**Why Start Here**: 
- Builds directly on your existing chain structure
- Computationally light
- Strong theoretical grounding (humans use minimal expressions for established entities)

---

### **Phase 3: Tier 2 - MTA (Modification Type Analysis)**
*Complexity: Medium | Expected Impact: Medium*

**Core Question**: *How* are entities modified? What kinds of linguistic devices are used?
```
For each mention in all chains:
  Analyze syntactic structure:
    ├─ Count adjective modifiers (amod dependencies)
    ├─ Count prepositional phrase modifiers (prep/pobj)
    ├─ Count relative clauses (relcl)
    ├─ Count compound modifiers (compound)
    └─ Calculate type diversity (entropy over categories)
```

**Features to Extract** (5 features):
1. `adjective_modification_rate`: Mentions with adjectives / total mentions
2. `prepositional_modification_rate`: Mentions with PP modifiers / total
3. `relative_clause_rate`: Mentions with relative clauses / total
4. `modification_type_entropy`: Shannon entropy over modification types
5. `avg_modifiers_per_mention`: Mean modification complexity

**Why Second**: 
- Requires dependency parsing (already available in spaCy)
- Captures stylistic differences in descriptive choices
- Complements Tier 1 by examining *quality* of modifications

---

### **Phase 4: Tier 3 - CSO (Context-Sensitive Overspecification)**
*Complexity: High | Expected Impact: High*

**Core Question**: Is specificity calibrated to actual disambiguation needs?
```
For each mention:
  Assess discourse context:
    ├─ Count similar entities in context window (±3 sentences)
    ├─ Determine if disambiguation is needed:
    │   ├─ Multiple same-type entities? → need distinction
    │   └─ Unique in context? → minimal expression sufficient
    └─ Calculate overspecification:
        If mention is modified BUT context is unambiguous:
          → mark as overspecified
```

**Features to Extract** (4 features):
1. `context_ambiguity_score`: Mean entities per context window needing distinction
2. `overspec_in_clear_context`: % mentions modified despite unique context
3. `underspec_in_ambiguous_context`: % minimal mentions when disambiguation needed
4. `specificity_calibration_score`: Correlation between context ambiguity and modification

**Why Third**: 
- Most cognitively sophisticated
- Requires entity similarity detection (challenging!)
- Likely strongest signal for AI detection (humans excel at pragmatic calibration)

---

### **Phase 5: Tier 4 - UIO (Unique-Identification Overspecification)**
*Complexity: Low-Medium | Expected Impact: Medium*

**Core Question**: Are singleton entities (mentioned once) over-described?
```
For each chain with length = 1 (singletons in our FastCoref data):
  Analyze the single mention:
    ├─ Count total tokens in mention
    ├─ Count modifiers applied
    ├─ Compare to baseline "minimal sufficient" expression
    └─ Calculate unnecessary elaboration
```

**Features to Extract** (3 features):
1. `singleton_avg_length`: Mean tokens in single-mention entities
2. `singleton_modification_rate`: % singletons with modifiers
3. `singleton_overelaboration_score`: Excess tokens beyond minimal identification

**Why Fourth**: 
- Straightforward to implement
- FastCoref gives us singletons naturally
- Tests cognitive economy principle directly

---

## Implementation Roadmap

### **Option A: Sequential Development** (Recommended)
Build and test each tier incrementally:
```
Week 1: Tier 1 RMO
  ├─ Implement 3 features
  ├─ Run ablation study (backbone + RMO)
  ├─ Assess F1 improvement
  └─ Decision: Continue if ΔF1 > 0.001

Week 2: Tier 2 MTA (if Tier 1 shows promise)
  ├─ Add 5 features
  ├─ Test backbone + RMO + MTA
  └─ Compare to backbone + MTA alone

Week 3: Tier 3 CSO (if cumulative improvement continues)
  ├─ Implement context analysis
  ├─ Add 4 features
  └─ Full model evaluation

Week 4: Tier 4 UIO + Final Integration
```

**Advantages**:
- Immediate feedback at each stage
- Can pivot if a tier underperforms
- Publications possible at each milestone

### **Option B: Parallel Development**
Build all tiers simultaneously, test together:
```
Week 1-2: Implement all 4 tiers
Week 3: Batch feature extraction
Week 4: Comprehensive ablation study
```

**Advantages**:
- Faster to complete
- Can analyze inter-tier interactions
- Better for hypothesis testing

---

## Feature Count Summary
```
Current Baseline:        6 features ✓
+ Tier 1 (RMO):         3 features → 9 total
+ Tier 2 (MTA):         5 features → 14 total  
+ Tier 3 (CSO):         4 features → 18 total
+ Tier 4 (UIO):         3 features → 21 total