import pandas as pd
import re
import spacy
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import numpy as np
import math
from tqdm import tqdm

class TimeMLAnnotator:
    """
    A comprehensive TimeML annotator for temporal reasoning in text.
    Handles TIMEX3, EVENT, and TLINK annotations according to TimeML specification.
    """
    
    def __init__(self):
        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            raise
        
        # Expanded and more sophisticated TimeML event indicators
        self.event_indicators = {
            'action': [
            'run', 'jump', 'build', 'create', 'make', 'do', 'perform', 'move', 'drive', 'walk', 'write', 'draw', 'cook', 'develop', 'launch', 'implement', 'execute', 'deliver', 'produce', 'construct', 'assemble', 'design', 'organize', 'plan', 'lead', 'manage', 'operate', 'travel', 'visit', 'explore', 'investigate', 'attack', 'defend', 'win', 'lose', 'compete', 'fight', 'play', 'work', 'study', 'learn', 'teach', 'train', 'practice'
            ],
            'state': [
            'be', 'have', 'exist', 'remain', 'stay', 'seem', 'appear', 'feel', 'look', 'sound', 'become', 'turn', 'grow', 'keep', 'hold', 'contain', 'possess', 'own', 'belong', 'represent', 'stand', 'lie', 'rest', 'wait', 'depend', 'consist', 'include', 'lack', 'need', 'require', 'prefer', 'like', 'love', 'hate', 'enjoy', 'want', 'wish', 'hope', 'expect'
            ],
            'occurrence': [
            'happen', 'occur', 'take place', 'arise', 'emerge', 'result', 'follow', 'ensue', 'transpire', 'develop', 'come', 'appear', 'materialize', 'surface', 'unfold', 'manifest', 'break out', 'start', 'begin'
            ],
            'aspectual': [
            'begin', 'start', 'finish', 'end', 'continue', 'stop', 'cease', 'resume', 'proceed', 'pause', 'complete', 'terminate', 'commence', 'conclude', 'interrupt', 'suspend', 'extend', 'last', 'persist'
            ],
            'perception': [
            'see', 'hear', 'feel', 'notice', 'observe', 'watch', 'listen', 'smell', 'taste', 'detect', 'perceive', 'sense', 'spot', 'witness', 'recognize', 'discern', 'identify', 'realize'
            ],
            'reporting': [
            'say', 'tell', 'report', 'announce', 'declare', 'state', 'mention', 'claim', 'explain', 'describe', 'reveal', 'assert', 'admit', 'confess', 'inform', 'notify', 'comment', 'remark', 'write', 'publish', 'broadcast', 'tweet', 'post'
            ],
            'modal': [
            'can', 'could', 'may', 'might', 'must', 'should', 'will', 'would', 'shall', 'ought', 'need', 'dare', 'used to', 'be able to', 'be going to', 'have to', 'want to', 'wish to', 'intend to', 'plan to', 'expect to'
            ],
            'intention': [
            'intend', 'plan', 'aim', 'hope', 'expect', 'decide', 'choose', 'promise', 'agree', 'offer', 'propose', 'vow', 'swear', 'guarantee'
            ],
            'communication': [
            'call', 'email', 'text', 'message', 'communicate', 'contact', 'converse', 'discuss', 'debate', 'argue', 'negotiate', 'consult', 'interview', 'question', 'ask', 'answer', 'reply', 'respond'
            ],
            'cognition': [
            'think', 'know', 'believe', 'understand', 'realize', 'remember', 'forget', 'consider', 'imagine', 'suppose', 'assume', 'guess', 'estimate', 'judge', 'conclude', 'predict', 'anticipate', 'doubt'
            ],
            'emotion': [
            'love', 'hate', 'like', 'enjoy', 'prefer', 'fear', 'regret', 'miss', 'appreciate', 'admire', 'respect', 'despise', 'resent', 'envy', 'pity', 'sympathize', 'worry', 'care'
            ]
        }

        # Expanded and more sophisticated temporal expression patterns
        self.timex_patterns = [
            # ISO and numeric dates
            (r'\b\d{4}-\d{2}-\d{2}\b', 'DATE'),  # 2023-06-01
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 'DATE'),  # 06/01/2023 or 6-1-23
            (r'\b\d{1,2}(st|nd|rd|th)?\s+(of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b', 'DATE'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}\b', 'DATE'),
            (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b', 'DATE'),
            (r'\b\d{4}\b', 'DATE'),  # year only

            # Relative dates
            (r'\b(yesterday|today|tomorrow|tonight|tonite|now|currently|presently|right now|at the moment|at this moment|at present)\b', 'DATE'),
            (r'\b(last|next|this|previous|coming|following|past|earlier|later)\s+(week|month|year|quarter|semester|season|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|weekend)\b', 'DATE'),
            (r'\b(in|after|before|since|until|by|during|over|within|for)\s+(a|an|\d+)\s+(second|minute|hour|day|week|month|year|decade|century|millennium)s?\b', 'DURATION'),
            (r'\b(since|until|by|during|over|within|for)\s+(last|next|this|previous|coming|following|past|earlier|later)?\s*(week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?\b', 'DATE'),

            # Durations and periods
            (r'\b\d+\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?|decades?|centuries?|millennia?)\b', 'DURATION'),
            (r'\b(a|an|one|half)\s+(second|minute|hour|day|week|month|year|decade|century|millennium)\b', 'DURATION'),
            (r'\b(all|whole|entire)\s+(day|week|month|year|night|morning|afternoon|evening)\b', 'DURATION'),
            (r'\b(half|quarter|third|fourth)\s+(hour|day|week|month|year)\b', 'DURATION'),

            # Times of day
            (r'\b\d{1,2}:\d{2}(:\d{2})?\s*([APap][Mm])?\b', 'TIME'),  # 14:30, 2:30 PM
            (r'\b\d{1,2}\s*([APap][Mm])\b', 'TIME'),  # 2 PM
            (r'\b(noon|midnight|morning|afternoon|evening|night|dawn|dusk|sunrise|sunset)\b', 'TIME'),

            # Named holidays and seasons
            (r'\b(Christmas|Easter|Thanksgiving|Halloween|Hanukkah|Ramadan|Diwali|New Year(\'s)?( Eve)?|Valentine(\'s)? Day|Independence Day|Labor Day|Memorial Day|Veterans Day|Mother(\'s)? Day|Father(\'s)? Day)\b', 'DATE'),
            (r'\b(Spring|Summer|Autumn|Fall|Winter)\b', 'DATE'),

            # Weekdays and months
            (r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', 'DATE'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', 'DATE'),
            (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\b', 'DATE'),

            # Ranges and intervals
            (r'\b(from|between)\s+.+?\s+(to|and|until|through|till)\s+.+?\b', 'DATE'),
            (r'\b(earlier|later|sooner|recently|lately|eventually|immediately|shortly|promptly|soon|suddenly|instantly|momentarily|temporarily|permanently)\b', 'DATE'),

            # Ordinal temporal references
            (r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|previous|next|following|current|upcoming)\s+(day|week|month|year|quarter|semester|season)\b', 'DATE'),
        ]
        
        # Temporal relation types (TLINK types)
        self.tlink_types = [
            'BEFORE', 'AFTER', 'INCLUDES', 'IS_INCLUDED', 
            'DURING', 'SIMULTANEOUS', 'BEGINS', 'ENDS',
            'IBEFORE', 'IAFTER', 'IDENTITY'
        ]
        
        # Event counters for unique IDs
        self.event_counter = 0
        self.timex_counter = 0
        self.tlink_counter = 0
    
    def extract_temporal_expressions(self, text: str) -> List[Dict]:
        """Extract temporal expressions (TIMEX3) from text."""
        timexes = []
        used_spans = set()  # Track used spans to avoid duplicates
        
        for pattern, timex_type in self.timex_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                span = (match.start(), match.end())
                # Avoid overlapping matches
                if not any(start <= span[0] < end or start < span[1] <= end for start, end in used_spans):
                    self.timex_counter += 1
                    timex = {
                        'tid': f't{self.timex_counter}',
                        'type': timex_type,
                        'value': self._normalize_temporal_value(match.group(), timex_type),
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end()
                    }
                    timexes.append(timex)
                    used_spans.add(span)
        
        return timexes
    
    def extract_events(self, text: str) -> List[Dict]:
        """Extract events from text using NLP and predefined indicators."""
        doc = self.nlp(text)
        events = []
        
        for token in doc:
            # Check if token is a verb or in our event indicators
            is_event = (
                token.pos_ == 'VERB' or 
                token.lemma_.lower() in [word for category in self.event_indicators.values() for word in category] or
                (token.pos_ == 'NOUN' and token.dep_ in ['nsubj', 'dobj'])  # Event nominals
            )
            
            if is_event:
                self.event_counter += 1
                
                # Determine event class
                event_class = self._classify_event(token)
                
                # Determine tense and aspect
                tense, aspect = self._analyze_tense_aspect(token)
                
                event = {
                    'eid': f'e{self.event_counter}',
                    'class': event_class,
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'tense': tense,
                    'aspect': aspect,
                    'start': token.idx,
                    'end': token.idx + len(token.text)
                }
                events.append(event)
        
        return events
    
    def _classify_event(self, token) -> str:
        """Classify event into TimeML event classes."""
        lemma = token.lemma_.lower()
        
        for event_class, words in self.event_indicators.items():
            if lemma in words:
                return event_class.upper()
        
        # Default classification based on POS and dependencies
        if token.pos_ == 'VERB':
            return 'ACTION'
        elif token.pos_ == 'NOUN':
            return 'OCCURRENCE'
        else:
            return 'STATE'
    
    def _analyze_tense_aspect(self, token) -> Tuple[str, str]:
        """
        Analyze tense and aspect of events using spaCy token features and context.
        Returns (tense, aspect) as strings.
        """
        tense = 'NONE'
        aspect = 'NONE'

        # Use spaCy's morphological features for tense
        morph = token.morph
        tag = token.tag_
        lemma = token.lemma_.lower()

        # Tense detection
        if 'Tense=Past' in str(morph) or tag in ['VBD', 'VBN']:
            tense = 'PAST'
        elif 'Tense=Pres' in str(morph) or tag in ['VBZ', 'VBP', 'VBG']:
            tense = 'PRESENT'
        elif 'Tense=Fut' in str(morph) or lemma in ['will', 'shall', 'gonna', 'going']:
            tense = 'FUTURE'
        elif tag == 'VB':
            # Infinitive, but check for modal/future context
            for child in token.children:
                if child.lemma_ in ['will', 'shall', 'would', 'should', 'might', 'may', 'can', 'could', 'must']:
                    tense = 'FUTURE'
                    break

        # Aspect detection
        if 'Aspect=Perf' in str(morph) or (tag == 'VBN' and any(aux.lemma_ == 'have' for aux in token.head.children)):
            aspect = 'PERFECTIVE'
        if 'Aspect=Prog' in str(morph) or tag == 'VBG':
            aspect = 'PROGRESSIVE'
        # Check for perfect progressive (have been VBG)
        if tag == 'VBG' and any(aux.lemma_ == 'have' for aux in token.head.children) and any(aux.lemma_ == 'be' for aux in token.head.children):
            aspect = 'PERFECTIVE_PROGRESSIVE'

        # Modal verbs (will, would, can, etc.)
        if tense == 'NONE' and hasattr(token, 'head') and token.head.pos_ == 'AUX' and token.head.lemma_ in ['will', 'shall', 'would', 'should', 'might', 'may', 'can', 'could', 'must']:
            tense = 'FUTURE'

        # If still undetermined, check auxiliaries
        if tense == 'NONE':
            for aux in token.children:
                if aux.dep_ == 'aux' and aux.lemma_ in ['have']:
                    aspect = 'PERFECTIVE'
                if aux.dep_ == 'aux' and aux.lemma_ in ['be']:
                    aspect = 'PROGRESSIVE'
                if aux.dep_ == 'aux' and aux.lemma_ in ['will', 'shall']:
                    tense = 'FUTURE'

        return tense, aspect
    
    def _normalize_temporal_value(self, text: str, timex_type: str) -> str:
        """Normalize temporal expressions to ISO format when possible."""
        text = text.lower().strip()
        
        try:
            # Simple normalization examples
            if text == 'today':
                return datetime.now().strftime('%Y-%m-%d')
            elif text == 'yesterday':
                return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            elif text == 'tomorrow':
                return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            elif text == 'now':
                return datetime.now().isoformat()
        except Exception:
            pass
        
        # Return original text if no normalization possible
        return text
    
    def extract_temporal_links(self, events: List[Dict], timexes: List[Dict], text: str) -> List[Dict]:
        """Extract temporal links (TLINKs) between events and temporal expressions with improved heuristics."""
        tlinks = []
        doc = self.nlp(text)

        # Build event and timex spans for overlap/proximity checks
        event_spans = [(e['start'], e['end'], e) for e in events]
        timex_spans = [(t['start'], t['end'], t) for t in timexes]

        # 1. Event-Event TLINKs: Use tense, aspect, and order, but also look for explicit temporal connectives
        temporal_connectives = {
            'before': 'BEFORE',
            'after': 'AFTER',
            'when': 'SIMULTANEOUS',
            'while': 'SIMULTANEOUS',
            'until': 'BEFORE',
            'since': 'AFTER',
            'as soon as': 'BEFORE',
            'once': 'AFTER',
            'during': 'DURING'
        }
        # Find connectives in text and map to event pairs
        for i in range(len(events) - 1):
            event1 = events[i]
            event2 = events[i + 1]
            # Extract text between events
            between_text = text[event1['end']:event2['start']].lower()
            relation = None
            for conn, rel in temporal_connectives.items():
                if conn in between_text:
                    relation = rel
                    break
            # Fallback to tense/aspect/position
            if not relation:
                if event1['tense'] == 'PAST' and event2['tense'] == 'PRESENT':
                    relation = 'BEFORE'
                elif event1['tense'] == 'PRESENT' and event2['tense'] == 'PAST':
                    relation = 'AFTER'
                elif event1['start'] < event2['start']:
                    relation = 'BEFORE'
                else:
                    relation = 'SIMULTANEOUS'
            self.tlink_counter += 1
            tlinks.append({
                'lid': f'l{self.tlink_counter}',
                'eventInstanceID': event1['eid'],
                'relatedToEvent': event2['eid'],
                'relType': relation
            })

        # 2. Event-Timex TLINKs: Use overlap, proximity, and dependency parse for temporal modifiers
        for event in events:
            event_span = (event['start'], event['end'])
            # Find closest timex (by character distance)
            closest_timex = None
            min_dist = float('inf')
            for timex in timexes:
                dist = min(abs(event['start'] - timex['end']), abs(event['end'] - timex['start']))
                if dist < min_dist:
                    min_dist = dist
                    closest_timex = timex
            # Use dependency parse to check if event is governed by a temporal modifier
            event_token = None
            for token in doc:
                if token.idx == event['start']:
                    event_token = token
                    break
            linked_timex = None
            if event_token:
                for child in event_token.children:
                    if child.dep_ in ['npadvmod', 'advmod', 'obl', 'nmod']:
                        # Check if child overlaps with any timex
                        for timex in timexes:
                            if (child.idx >= timex['start'] and child.idx < timex['end']) or \
                               (timex['start'] >= child.idx and timex['start'] < child.idx + len(child.text)):
                                linked_timex = timex
                                break
            # Prefer dependency-linked timex, else closest
            final_timex = linked_timex if linked_timex else closest_timex
            if final_timex and min_dist < 100:  # More flexible threshold
                self.tlink_counter += 1
                tlinks.append({
                    'lid': f'l{self.tlink_counter}',
                    'eventInstanceID': event['eid'],
                    'relatedToTime': final_timex['tid'],
                    'relType': 'IS_INCLUDED' if min_dist < 30 else 'DURING'
                })

        # 3. Timex-Timex TLINKs: If two timexes are in a range (e.g., "from X to Y")
        for i in range(len(timexes) - 1):
            t1 = timexes[i]
            t2 = timexes[i + 1]
            between_text = text[t1['end']:t2['start']].lower()
            if 'to' in between_text or 'until' in between_text or 'through' in between_text:
                self.tlink_counter += 1
                tlinks.append({
                    'lid': f'l{self.tlink_counter}',
                    'eventInstanceID': t1['tid'],
                    'relatedToTime': t2['tid'],
                    'relType': 'BEFORE'
                })

        return tlinks
    
    def annotate_text(self, text: str) -> Dict:
        """Perform complete TimeML annotation of text."""
        # Extract temporal elements
        timexes = self.extract_temporal_expressions(text)
        events = self.extract_events(text)
        tlinks = self.extract_temporal_links(events, timexes, text)
        
        return {
            'text': text,
            'timexes': timexes,
            'events': events,
            'tlinks': tlinks
        }
    
    def annotate_dataframe(self, df: pd.DataFrame, text_column: str = 'generation') -> pd.DataFrame:
        """Annotate all texts in a dataframe column."""
        results = []
        
        for idx, row in df.iterrows():
            text = str(row[text_column])  # Convert to string to handle any data type
            annotation = self.annotate_text(text)
            
            # Add row index for tracking
            annotation['row_index'] = idx
            results.append(annotation)
        
        # Create new dataframe with annotations
        annotated_df = df.copy()
        annotated_df['timeml_annotation'] = results
        
        return annotated_df
    
    def create_temporal_graph(self, annotations: List[Dict]) -> nx.DiGraph:
        """Create a temporal graph from TimeML annotations."""
        G = nx.DiGraph()
        
        for annotation in annotations:
            # Add nodes for events and temporal expressions
            for event in annotation['events']:
                G.add_node(event['eid'], 
                          type='event', 
                          text=event['text'],
                          event_class=event['class'],  # Fixed attribute name
                          tense=event['tense'])
            
            for timex in annotation['timexes']:
                G.add_node(timex['tid'], 
                          type='timex', 
                          text=timex['text'],
                          timex_type=timex['type'],
                          value=timex['value'])
            
            # Add edges for temporal links
            for tlink in annotation['tlinks']:
                if 'relatedToEvent' in tlink:
                    G.add_edge(tlink['eventInstanceID'], 
                              tlink['relatedToEvent'],
                              relation=tlink['relType'])
                elif 'relatedToTime' in tlink:
                    G.add_edge(tlink['eventInstanceID'], 
                              tlink['relatedToTime'],
                              relation=tlink['relType'])
        
        return G
    
    def visualize_temporal_graph(self, G: nx.DiGraph, figsize: Tuple[int, int] = (12, 8)):
        """Visualize the temporal graph."""
        if G.number_of_nodes() == 0:
            print("No nodes to visualize in the graph.")
            return
            
        plt.figure(figsize=figsize)
        
        # Separate event and timex nodes
        event_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'event']
        timex_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'timex']
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        if event_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=event_nodes, 
                                  node_color='lightblue', node_size=800, label='Events')
        if timex_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=timex_nodes, 
                                  node_color='lightcoral', node_size=800, label='Temporal Expressions')
        
        # Draw edges with different colors for different relations
        edge_colors = []
        for u, v, d in G.edges(data=True):
            if d['relation'] in ['BEFORE', 'AFTER']:
                edge_colors.append('red')
            elif d['relation'] in ['INCLUDES', 'IS_INCLUDED', 'DURING']:
                edge_colors.append('blue')
            else:
                edge_colors.append('gray')
        
        if edge_colors:
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=20)
        
        # Add labels
        labels = {n: d['text'][:10] + '...' if len(d['text']) > 10 else d['text'] 
                 for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Add edge labels for relations
        if G.number_of_edges() > 0:
            edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        plt.title('Temporal Graph from TimeML Annotations')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def export_to_timeml_xml(self, annotation: Dict) -> str:
        """Export annotation to TimeML XML format."""
        root = ET.Element('TimeML')
        
        # Add original text
        text_elem = ET.SubElement(root, 'TEXT')
        text_elem.text = annotation['text']
        
        # Add TIMEX3 elements
        for timex in annotation['timexes']:
            timex_elem = ET.SubElement(root, 'TIMEX3')
            timex_elem.set('tid', timex['tid'])
            timex_elem.set('type', timex['type'])
            timex_elem.set('value', timex['value'])
            timex_elem.text = timex['text']
        
        # Add EVENT elements
        for event in annotation['events']:
            event_elem = ET.SubElement(root, 'EVENT')
            event_elem.set('eid', event['eid'])
            event_elem.set('class', event['class'])
            event_elem.set('tense', event['tense'])
            event_elem.set('aspect', event['aspect'])
            event_elem.text = event['text']
        
        # Add TLINK elements
        for tlink in annotation['tlinks']:
            tlink_elem = ET.SubElement(root, 'TLINK')
            tlink_elem.set('lid', tlink['lid'])
            tlink_elem.set('eventInstanceID', tlink['eventInstanceID'])
            if 'relatedToEvent' in tlink:
                tlink_elem.set('relatedToEvent', tlink['relatedToEvent'])
            elif 'relatedToTime' in tlink:
                tlink_elem.set('relatedToTime', tlink['relatedToTime'])
            tlink_elem.set('relType', tlink['relType'])
        
        return ET.tostring(root, encoding='unicode')

    def get_annotation_statistics(self, annotations: List[Dict]) -> Dict:
        """Get comprehensive statistics about the annotations."""
        stats = {
            'total_texts': len(annotations),
            'total_events': 0,
            'total_timexes': 0,
            'total_tlinks': 0,
            'event_classes': defaultdict(int),
            'timex_types': defaultdict(int),
            'tlink_types': defaultdict(int),
            'avg_events_per_text': 0,
            'avg_timexes_per_text': 0,
            'avg_tlinks_per_text': 0
        }
        
        for annotation in annotations:
            stats['total_events'] += len(annotation['events'])
            stats['total_timexes'] += len(annotation['timexes'])
            stats['total_tlinks'] += len(annotation['tlinks'])
            
            for event in annotation['events']:
                stats['event_classes'][event['class']] += 1
            
            for timex in annotation['timexes']:
                stats['timex_types'][timex['type']] += 1
            
            for tlink in annotation['tlinks']:
                stats['tlink_types'][tlink['relType']] += 1
        
        if stats['total_texts'] > 0:
            stats['avg_events_per_text'] = stats['total_events'] / stats['total_texts']
            stats['avg_timexes_per_text'] = stats['total_timexes'] / stats['total_texts']
            stats['avg_tlinks_per_text'] = stats['total_tlinks'] / stats['total_texts']
        
        return stats
    

# =========== Enhanced Temporal Granularity Detection =========    
 
class ImprovedTemporalGranularityExtractor:
    """
    Enhanced temporal granularity detection with much better coverage
    """

    def __init__(self):
        # Expanded granularity levels with more precise mapping
        self.granularity_levels = {
            # Very long term
            'millennium': 6, 'millennia': 6, 'millenia': 6,
            'century': 5, 'centuries': 5,
            'decade': 4, 'decades': 4,

            # Medium term
            'year': 3, 'years': 3, 'yearly': 3, 'annual': 3, 'annually': 3,
            'month': 2, 'months': 2, 'monthly': 2,
            'week': 1, 'weeks': 1, 'weekly': 1,

            # Short term
            'day': 0, 'days': 0, 'daily': 0,
            'hour': -1, 'hours': -1, 'hourly': -1,
            'minute': -2, 'minutes': -2, 'min': -2, 'mins': -2,
            'second': -3, 'seconds': -3, 'sec': -3, 'secs': -3,

            # Instant
            'instant': -4, 'moment': -4, 'immediately': -4, 'now': -4
        }

        # Enhanced pattern-based detection
        self.granularity_patterns = [
            # Years - various formats
            (r'\b\d{4}\b(?!\s*[:\-/])', 3),  # 2023 (not followed by time separators)
            (r'\b\d{4}s\b', 4),  # 1990s (decade)
            (r'\b\d{2}th century\b', 5),  # 20th century
            (r'\b\d{1,2}(st|nd|rd|th)\s+century\b', 5),  # 1st century

            # Dates - month precision
            (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b', 2),
            (r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+\d{4}\b', 2),
            (r'\b\d{1,2}[/-]\d{4}\b', 2),  # MM/YYYY or MM-YYYY

            # Full dates - day precision
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 0),  # MM/DD/YYYY
            (r'\b\d{4}-\d{2}-\d{2}\b', 0),  # ISO date
            (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}\b', 0),

            # Times - hour/minute/second precision
            (r'\b\d{1,2}:\d{2}\s*([ap]m)?\b', -1),  # HH:MM format (hour precision)
            (r'\b\d{1,2}:\d{2}:\d{2}\b', -2),  # HH:MM:SS format (minute precision)
            (r'\b\d{1,2}\s*([ap]m)\b', -1),  # 3 PM format

            # Week-based expressions
            (r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 1),
            (r'\bweekend\b', 1),

            # Relative temporal expressions with implied granularity
            (r'\b(today|yesterday|tomorrow|tonight)\b', 0),  # Day precision
            (r'\b(this|next|last)\s+(week|weekend)\b', 1),  # Week precision
            (r'\b(this|next|last)\s+(month)\b', 2),  # Month precision
            (r'\b(this|next|last)\s+(year)\b', 3),  # Year precision
            (r'\b(this|next|last)\s+(decade)\b', 4),  # Decade precision
            (r'\b(this|next|last)\s+(century)\b', 5),  # Century precision

            # Duration expressions (infer granularity from unit)
            (r'\b\d+\s*(milliseconds?|ms)\b', -4),  # Instant
            (r'\b\d+\s*(seconds?|secs?|s)\b', -3),  # Second
            (r'\b\d+\s*(minutes?|mins?|m)\b', -2),  # Minute
            (r'\b\d+\s*(hours?|hrs?|h)\b', -1),  # Hour
            (r'\b\d+\s*(days?|d)\b', 0),  # Day
            (r'\b\d+\s*(weeks?|wks?|w)\b', 1),  # Week
            (r'\b\d+\s*(months?|mos?)\b', 2),  # Month
            (r'\b\d+\s*(years?|yrs?|y)\b', 3),  # Year
            (r'\b\d+\s*(decades?)\b', 4),  # Decade

            # Fractional expressions
            (r'\bhalf\s+(hour|hr)\b', -1),
            (r'\bhalf\s+(day)\b', 0),
            (r'\bhalf\s+(week)\b', 1),
            (r'\bhalf\s+(month)\b', 2),
            (r'\bhalf\s+(year)\b', 3),
            (r'\bquarter\s+(hour|hr)\b', -1),

            # Seasonal/cyclical (month-level precision)
            (r'\b(spring|summer|fall|autumn|winter)\b', 2),
            (r'\b(quarterly|semester)\b', 2),

            # Historical periods (year-level precision)
            (r'\b(ancient|medieval|renaissance|modern|contemporary)\b', 3),

            # Event-based temporal references (day precision)
            (r'\b(birthday|anniversary|holiday|graduation|wedding)\b', 0),

            # Business/academic time (various precisions)
            (r'\b(semester|quarter|term)\b', 2),  # Month-level
            (r'\b(fiscal\s+year|academic\s+year)\b', 3),  # Year-level
            (r'\bdeadline\b', 0),  # Usually day-specific
        ]

        # Contextual clues that might indicate granularity even without explicit temporal words
        self.contextual_patterns = [
            # Age-related (year precision)
            (r'\b\d{1,2}\s+years?\s+old\b', 3),
            (r'\bage\s+\d{1,2}\b', 3),

            # Event frequency (infer from context)
            (r'\b(daily|everyday)\b', 0),
            (r'\b(weekly|every\s+week)\b', 1),
            (r'\b(monthly|every\s+month)\b', 2),
            (r'\b(yearly|annually|every\s+year)\b', 3),

            # Urgency indicators (shorter granularity)
            (r'\b(urgent|immediately|asap|right\s+now|instantly)\b', -4),
            (r'\b(soon|shortly|quickly|rapidly)\b', -1),

            # Planning horizons (longer granularity)
            (r'\b(long[\-\s]term|strategic|future\s+planning)\b', 3),
            (r'\b(short[\-\s]term|immediate\s+term)\b', 0),
        ]

    def _determine_enhanced_granularity(self, timex_text: str, full_text: str = "",
                                       context_window: int = 100) -> List[int]:
        """
        Enhanced granularity detection with multiple strategies
        """
        granularities = []
        text_lower = timex_text.lower()

        # Strategy 1: Direct keyword matching
        for keyword, level in self.granularity_levels.items():
            if keyword in text_lower:
                granularities.append(level)

        # Strategy 2: Pattern-based detection
        for pattern, level in self.granularity_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                granularities.append(level)

        # Strategy 3: Contextual analysis
        if full_text and timex_text in full_text:
            timex_pos = full_text.lower().find(text_lower)
            if timex_pos >= 0:
                start_pos = max(0, timex_pos - context_window)
                end_pos = min(len(full_text), timex_pos + len(timex_text) + context_window)
                context = full_text[start_pos:end_pos].lower()

                for pattern, level in self.contextual_patterns:
                    if re.search(pattern, context, re.IGNORECASE):
                        granularities.append(level)

        # Strategy 4: Fallback heuristics
        if not granularities:
            granularities.extend(self._fallback_granularity_detection(text_lower))

        # Remove duplicates while preserving order
        seen = set()
        unique_granularities = []
        for g in granularities:
            if g not in seen:
                seen.add(g)
                unique_granularities.append(g)

        return unique_granularities

    def _fallback_granularity_detection(self, text_lower: str) -> List[int]:
        """Fallback detection for edge cases"""
        granularities = []

        # Numeric-only patterns
        if re.search(r'^\d+$', text_lower.strip()):
            if re.search(r'^\d{4}$', text_lower.strip()):
                granularities.append(3)  # Year
            elif re.search(r'^\d{1,2}$', text_lower.strip()):
                granularities.append(0)  # Probably day

        # Ordinal patterns
        if re.search(r'\d+(st|nd|rd|th)', text_lower):
            granularities.append(0)  # Usually day precision

        # Time-like patterns
        if ':' in text_lower or 'am' in text_lower or 'pm' in text_lower:
            granularities.append(-1)  # Hour precision

        # Default fallback
        if not granularities:
            if len(text_lower) <= 3:
                granularities.append(0)  # Short expressions usually day-level
            else:
                granularities.append(2)  # Longer expressions often month-level

        return granularities

    def calculate_improved_temporal_granularity_diversity(self, annotation: Dict) -> float:
        """Calculate temporal granularity diversity with improved coverage"""
        if not annotation.get('timexes'):
            return 0.0

        all_granularities = []
        full_text = annotation.get('text', '')

        # Extract granularities from all temporal expressions
        for timex in annotation['timexes']:
            timex_text = timex['text']
            detected_granularities = self._determine_enhanced_granularity(
                timex_text, full_text, context_window=150
            )
            all_granularities.extend(detected_granularities)

        # If no granularities detected, try extracting from events
        if not all_granularities and annotation.get('events'):
            for event in annotation['events']:
                if event.get('tense'):
                    if event['tense'] in ['PAST', 'PRESENT']:
                        all_granularities.append(0)  # Often day-level
                    elif event['tense'] == 'FUTURE':
                        all_granularities.append(1)  # Often week-level or longer

        if not all_granularities:
            return 0.0

        # Calculate entropy
        granularity_counts = Counter(all_granularities)
        total = len(all_granularities)

        if total <= 1:
            return 0.0

        # Shannon entropy calculation
        entropy = 0.0
        for count in granularity_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by theoretical maximum entropy
        unique_levels = len(granularity_counts)
        max_entropy = math.log2(unique_levels) if unique_levels > 1 else 1.0

        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Apply bonus for having multiple different granularity levels
        diversity_bonus = min(unique_levels / 5.0, 1.0)  # Bonus up to 5 different levels

        # Combine entropy and diversity bonus
        final_score = (normalized_entropy * 0.7 + diversity_bonus * 0.3)

        return min(final_score, 1.0)
    
# ======================= Advanced Features =======================

"""
Advanced Temporal Reasoning Features for AI Detection - 
Based on  existing TimeML annotator, this extends it with additional features
that capture sophisticated temporal reasoning patterns.
"""

class AdvancedTemporalFeatureExtractor:
    """
    Advanced temporal reasoning feature extractor for AI detection.
    Builds on TimeML annotations to extract sophisticated temporal patterns.
    """

    def __init__(self, timeml_annotator):
        """
        Initialize with your existing TimeML annotator

        Parameters:
        -----------
        timeml_annotator : TimeMLAnnotator
            Your existing TimeML annotator instance
        """
        self.timeml_annotator = timeml_annotator

        # Feature names that will be extracted
        self.feature_names = [
            'temporal_density',
            'temporal_complexity_score',
            'temporal_coherence',
            'temporal_granularity_diversity',
            'temporal_perspective_consistency',
            'temporal_anchoring_strength',
            'temporal_narrative_flow',
            'temporal_disambiguation_quality',
            'temporal_reference_precision',
            'temporal_logical_consistency',
            'temporal_semantic_richness',
            'temporal_syntactic_integration'
        ]

        # Temporal granularity levels - UPDATED
        self.granularity_levels = {
            'millennium': 6, 'millennia': 6,
            'century': 5, 'centuries': 5,
            'decade': 4, 'decades': 4,
            'year': 3, 'years': 3, 'yearly': 3, 'annual': 3,
            'month': 2, 'months': 2, 'monthly': 2,
            'week': 1, 'weeks': 1, 'weekly': 1,
            'day': 0, 'days': 0, 'daily': 0,
            'hour': -1, 'hours': -1, 'hourly': -1,
            'minute': -2, 'minutes': -2,
            'second': -3, 'seconds': -3,
            'instant': -4, 'moment': -4, 'immediately': -4, 'now': -4
        }

        # Temporal perspective markers
        self.perspective_markers = {
            'past': ['was', 'were', 'had', 'did', 'went', 'came', 'saw', 'said', 'told', 'happened', 'occurred'],
            'present': ['is', 'are', 'am', 'has', 'have', 'do', 'does', 'go', 'come', 'see', 'say', 'tell'],
            'future': ['will', 'shall', 'going to', 'would', 'should', 'might', 'may', 'could', 'can']
        }

        # Temporal connectives for flow analysis
        self.temporal_connectives = {
            'sequence': ['then', 'next', 'after', 'afterwards', 'subsequently', 'following', 'later'],
            'simultaneity': ['while', 'when', 'as', 'during', 'meanwhile', 'simultaneously', 'at the same time'],
            'causality': ['because', 'since', 'as a result', 'therefore', 'thus', 'consequently', 'due to'],
            'contrast': ['however', 'but', 'although', 'despite', 'nevertheless', 'nonetheless', 'yet'],
            'duration': ['for', 'during', 'throughout', 'over', 'across', 'within', 'until', 'since']
        }

        # Initialize improved granularity extractor
        self.granularity_extractor = ImprovedTemporalGranularityExtractor()

        print("✅ Advanced Temporal Feature Extractor initialized")
        print(f"📊 Will extract {len(self.feature_names)} temporal reasoning features")

    def _calculate_temporal_density(self, annotation: Dict) -> float:
        """Calculate the density of temporal elements relative to text length"""
        text_length = len(annotation['text'].split())
        if text_length == 0:
            return 0.0

        total_temporal_elements = len(annotation['timexes']) + len(annotation['events'])
        return total_temporal_elements / text_length

    def _calculate_temporal_complexity_score(self, annotation: Dict) -> float:
        """Calculate overall temporal complexity based on multiple factors"""
        if not annotation['events'] and not annotation['timexes']:
            return 0.0

        # Base complexity from element counts
        event_count = len(annotation['events'])
        timex_count = len(annotation['timexes'])
        tlink_count = len(annotation['tlinks'])

        # Weighted complexity score
        complexity = (event_count * 1.0 + timex_count * 1.5 + tlink_count * 2.0)

        # Add complexity from event class diversity
        event_classes = set(event['class'] for event in annotation['events'])
        class_diversity = len(event_classes) / max(event_count, 1)

        # Add complexity from temporal expression types
        timex_types = set(timex['type'] for timex in annotation['timexes'])
        type_diversity = len(timex_types) / max(timex_count, 1)

        # Add complexity from tense variations
        tenses = set(event['tense'] for event in annotation['events'])
        tense_diversity = len(tenses) / max(event_count, 1)

        # Normalize by text length
        text_length = len(annotation['text'].split())
        normalized_complexity = complexity / max(text_length / 10, 1)

        return normalized_complexity * (1 + class_diversity + type_diversity + tense_diversity)

    def _calculate_temporal_coherence(self, annotation: Dict) -> float:
        """Calculate temporal coherence based on consistency of temporal relationships"""
        if len(annotation['tlinks']) < 2:
            return 0.0

        # Build temporal relation graph
        G = nx.DiGraph()

        # Add temporal links as edges
        for tlink in annotation['tlinks']:
            if 'relatedToEvent' in tlink:
                G.add_edge(tlink['eventInstanceID'], tlink['relatedToEvent'],
                          relation=tlink['relType'])
            elif 'relatedToTime' in tlink:
                G.add_edge(tlink['eventInstanceID'], tlink['relatedToTime'],
                          relation=tlink['relType'])

        if G.number_of_nodes() < 2:
            return 0.0

        # Calculate coherence metrics
        coherence_score = 0.0

        # 1. Connectivity - how well connected is the temporal network?
        if G.number_of_nodes() > 0:
            connectivity = G.number_of_edges() / G.number_of_nodes()
            coherence_score += min(connectivity, 1.0) * 0.4

        # 2. Consistency - check for contradictory relations
        contradictions = 0
        total_paths = 0

        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 != node2:
                    try:
                        paths = list(nx.all_simple_paths(G, node1, node2, cutoff=3))
                        if len(paths) > 1:
                            total_paths += 1
                            # Check for contradictory paths (simplified)
                            relations = []
                            for path in paths:
                                path_relations = []
                                for i in range(len(path) - 1):
                                    if G.has_edge(path[i], path[i+1]):
                                        path_relations.append(G[path[i]][path[i+1]]['relation'])
                                relations.append(path_relations)

                            # Simple contradiction check
                            if len(set(str(r) for r in relations)) > 1:
                                contradictions += 1
                    except:
                        continue

        if total_paths > 0:
            consistency = 1.0 - (contradictions / total_paths)
            coherence_score += consistency * 0.6

        return coherence_score

    def _calculate_temporal_granularity_diversity(self, annotation: Dict) -> float:
        """Calculate temporal granularity diversity - NOW USING IMPROVED VERSION"""
        return self.granularity_extractor.calculate_improved_temporal_granularity_diversity(annotation)

    def _determine_granularity(self, timex_text: str) -> Optional[int]:
        """Determine the granularity level of a temporal expression - LEGACY METHOD"""
        # This is kept for backward compatibility but the improved version is used
        return self.granularity_extractor._determine_enhanced_granularity(timex_text, "")[0] if self.granularity_extractor._determine_enhanced_granularity(timex_text, "") else None

    def _calculate_temporal_perspective_consistency(self, annotation: Dict) -> float:
        """Calculate consistency of temporal perspective throughout the text"""
        if not annotation['events']:
            return 0.0

        # Count tense usage
        tense_counts = Counter(event['tense'] for event in annotation['events'])
        total_events = len(annotation['events'])

        if total_events == 0:
            return 0.0

        # Calculate dominant tense proportion
        dominant_tense_count = max(tense_counts.values())
        consistency = dominant_tense_count / total_events

        # Penalize excessive tense switching
        tense_switches = 0
        prev_tense = None

        for event in annotation['events']:
            if prev_tense is not None and event['tense'] != prev_tense:
                tense_switches += 1
            prev_tense = event['tense']

        switch_penalty = tense_switches / max(total_events - 1, 1)
        adjusted_consistency = consistency * (1 - switch_penalty * 0.5)

        return min(adjusted_consistency, 1.0)

    def _calculate_temporal_anchoring_strength(self, annotation: Dict) -> float:
        """Calculate how well temporal events are anchored to specific time points"""
        if not annotation['events'] or not annotation['timexes']:
            return 0.0

        # Count events with temporal anchors (via TLINKs)
        anchored_events = set()

        for tlink in annotation['tlinks']:
            if 'relatedToTime' in tlink:
                anchored_events.add(tlink['eventInstanceID'])

        anchoring_ratio = len(anchored_events) / len(annotation['events'])

        # Bonus for precise temporal anchors
        precise_anchors = 0
        for timex in annotation['timexes']:
            if self._is_precise_temporal_expression(timex['text']):
                precise_anchors += 1

        precision_bonus = precise_anchors / max(len(annotation['timexes']), 1)

        return anchoring_ratio * (1 + precision_bonus)

    def _is_precise_temporal_expression(self, timex_text: str) -> bool:
        """Check if a temporal expression is precise (specific date/time)"""
        # ISO dates, specific dates, specific times
        precise_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # ISO date
            r'\d{1,2}:\d{2}',      # Time
            r'\d{1,2}/\d{1,2}/\d{4}',  # Date
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
        ]

        for pattern in precise_patterns:
            if re.search(pattern, timex_text.lower()):
                return True

        return False

    def _calculate_temporal_narrative_flow(self, annotation: Dict) -> float:
        """Calculate the quality of temporal narrative flow using connectives"""
        if not annotation['events']:
            return 0.0

        text = annotation['text'].lower()
        flow_score = 0.0

        # Count temporal connectives
        connective_counts = {}
        total_connectives = 0

        for category, connectives in self.temporal_connectives.items():
            count = sum(1 for conn in connectives if conn in text)
            connective_counts[category] = count
            total_connectives += count

        if total_connectives == 0:
            return 0.0

        # Calculate diversity of connective types
        used_categories = sum(1 for count in connective_counts.values() if count > 0)
        diversity = used_categories / len(self.temporal_connectives)

        # Calculate density relative to events
        density = total_connectives / len(annotation['events'])

        # Balance between diversity and density
        flow_score = (diversity * 0.6 + min(density, 1.0) * 0.4)

        return flow_score

    def _calculate_temporal_disambiguation_quality(self, annotation: Dict) -> float:
        """Calculate how well temporal ambiguities are resolved"""
        if not annotation['timexes']:
            return 0.0

        # Count ambiguous vs. specific temporal expressions
        ambiguous_count = 0
        specific_count = 0

        ambiguous_patterns = [
            r'\b(recently|lately|soon|eventually|earlier|later|now|today|yesterday|tomorrow)\b',
            r'\b(this|next|last|previous)\s+(week|month|year|time)\b',
            r'\b(in the (past|future))\b'
        ]

        for timex in annotation['timexes']:
            text = timex['text'].lower()
            is_ambiguous = any(re.search(pattern, text) for pattern in ambiguous_patterns)

            if is_ambiguous:
                ambiguous_count += 1
            else:
                specific_count += 1

        total_timexes = len(annotation['timexes'])

        # Higher score for more specific expressions
        disambiguation_score = specific_count / total_timexes

        # Bonus for having normalization values
        normalized_count = sum(1 for timex in annotation['timexes']
                             if timex.get('value', timex['text']) != timex['text'])

        normalization_bonus = normalized_count / total_timexes

        return disambiguation_score * (1 + normalization_bonus)

    def _calculate_temporal_reference_precision(self, annotation: Dict) -> float:
        """Calculate precision of temporal references"""
        if not annotation['timexes']:
            return 0.0

        precision_score = 0.0

        for timex in annotation['timexes']:
            text = timex['text'].lower()

            # Score based on specificity
            if re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', text):  # ISO datetime
                precision_score += 1.0
            elif re.search(r'\d{1,2}:\d{2}(:\d{2})?', text):  # Time
                precision_score += 0.9
            elif re.search(r'\d{1,2}/\d{1,2}/\d{4}', text):  # Date
                precision_score += 0.8
            elif re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}', text):  # Month + day
                precision_score += 0.7
            elif re.search(r'\d{4}', text):  # Year
                precision_score += 0.5
            elif text in ['today', 'yesterday', 'tomorrow']:
                precision_score += 0.6
            else:
                precision_score += 0.2

        return precision_score / len(annotation['timexes'])

    def _calculate_temporal_logical_consistency(self, annotation: Dict) -> float:
        """Calculate logical consistency of temporal relationships"""
        if len(annotation['tlinks']) < 2:
            return 0.0

        # Build temporal constraint graph
        constraints = []

        for tlink in annotation['tlinks']:
            if 'relatedToEvent' in tlink:
                source = tlink['eventInstanceID']
                target = tlink['relatedToEvent']
                relation = tlink['relType']
                constraints.append((source, target, relation))

        if len(constraints) < 2:
            return 0.0

        # Simple consistency check
        consistent_count = 0
        total_checks = 0

        # Check for direct contradictions
        for i, (s1, t1, r1) in enumerate(constraints):
            for j, (s2, t2, r2) in enumerate(constraints):
                if i != j:
                    total_checks += 1

                    # Check for contradictory relations
                    if s1 == s2 and t1 == t2 and r1 != r2:
                        # Direct contradiction
                        continue
                    elif s1 == t2 and t1 == s2:
                        # Inverse relations should be consistent
                        inverse_consistent = self._check_inverse_consistency(r1, r2)
                        if inverse_consistent:
                            consistent_count += 1
                    else:
                        consistent_count += 1

        return consistent_count / max(total_checks, 1)

    def _check_inverse_consistency(self, rel1: str, rel2: str) -> bool:
        """Check if two relations are consistent inverses"""
        inverse_pairs = [
            ('BEFORE', 'AFTER'),
            ('INCLUDES', 'IS_INCLUDED'),
            ('BEGINS', 'BEGUN_BY'),
            ('ENDS', 'ENDED_BY')
        ]

        for pair in inverse_pairs:
            if (rel1, rel2) == pair or (rel2, rel1) == pair:
                return True

        return rel1 == rel2  # Same relation is consistent

    def _calculate_temporal_semantic_richness(self, annotation: Dict) -> float:
        """Calculate semantic richness of temporal expressions and events"""
        if not annotation['events'] and not annotation['timexes']:
            return 0.0

        richness_score = 0.0

        # Event semantic richness
        if annotation['events']:
            event_classes = set(event['class'] for event in annotation['events'])
            event_aspects = set(event['aspect'] for event in annotation['events'])

            class_diversity = len(event_classes) / len(annotation['events'])
            aspect_diversity = len(event_aspects) / len(annotation['events'])

            richness_score += (class_diversity + aspect_diversity) / 2

        # Temporal expression semantic richness
        if annotation['timexes']:
            timex_types = set(timex['type'] for timex in annotation['timexes'])
            type_diversity = len(timex_types) / len(annotation['timexes'])

            # Semantic complexity of temporal expressions
            semantic_complexity = 0
            for timex in annotation['timexes']:
                text = timex['text'].lower()
                if any(word in text for word in ['during', 'throughout', 'within', 'between']):
                    semantic_complexity += 1

            semantic_bonus = semantic_complexity / len(annotation['timexes'])
            richness_score += (type_diversity + semantic_bonus) / 2

        return richness_score

    def _calculate_temporal_syntactic_integration(self, annotation: Dict) -> float:
        """Calculate how well temporal elements are syntactically integrated"""
        if not annotation['events'] and not annotation['timexes']:
            return 0.0

        text = annotation['text']
        total_elements = len(annotation['events']) + len(annotation['timexes'])

        # Count temporal elements that are syntactically integrated
        integrated_count = 0

        for event in annotation['events']:
            start = event['start']
            end = event['end']

            # Check context around event
            before_text = text[max(0, start-20):start].lower()
            after_text = text[end:end+20].lower()

            # Simple integration check
            if any(word in before_text for word in ['the', 'a', 'an', 'this', 'that', 'has', 'had', 'will', 'would']):
                integrated_count += 1
            elif any(word in after_text for word in ['the', 'a', 'an', 'and', 'or', 'but', 'that', 'which']):
                integrated_count += 1

        for timex in annotation['timexes']:
            start = timex['start']
            end = timex['end']

            # Check context around timex
            before_text = text[max(0, start-20):start].lower()
            after_text = text[end:end+20].lower()

            # Simple integration check
            if any(word in before_text for word in ['on', 'at', 'in', 'during', 'before', 'after', 'since', 'until']):
                integrated_count += 1
            elif any(word in after_text for word in ['the', 'a', 'an', 'and', 'or', 'but']):
                integrated_count += 1

        return integrated_count / total_elements if total_elements > 0 else 0.0

    def extract_temporal_features(self, text: str) -> Dict[str, float]:
        """Extract all temporal reasoning features from text"""
        features = {name: 0.0 for name in self.feature_names}

        if not text or len(text.strip()) < 10:
            return features

        try:
            # Get TimeML annotation
            annotation = self.timeml_annotator.annotate_text(text)

            # Extract features
            features['temporal_density'] = self._calculate_temporal_density(annotation)
            features['temporal_complexity_score'] = self._calculate_temporal_complexity_score(annotation)
            features['temporal_coherence'] = self._calculate_temporal_coherence(annotation)
            features['temporal_granularity_diversity'] = self._calculate_temporal_granularity_diversity(annotation)
            features['temporal_perspective_consistency'] = self._calculate_temporal_perspective_consistency(annotation)
            features['temporal_anchoring_strength'] = self._calculate_temporal_anchoring_strength(annotation)
            features['temporal_narrative_flow'] = self._calculate_temporal_narrative_flow(annotation)
            features['temporal_disambiguation_quality'] = self._calculate_temporal_disambiguation_quality(annotation)
            features['temporal_reference_precision'] = self._calculate_temporal_reference_precision(annotation)
            features['temporal_logical_consistency'] = self._calculate_temporal_logical_consistency(annotation)
            features['temporal_semantic_richness'] = self._calculate_temporal_semantic_richness(annotation)
            features['temporal_syntactic_integration'] = self._calculate_temporal_syntactic_integration(annotation)

        except Exception as e:
            print(f"⚠️ Error extracting temporal features: {e}")

        return features

    def extract_temporal_features_batch(self, df: pd.DataFrame, text_column: str = 'generation') -> pd.DataFrame:
        """Extract temporal features for entire dataframe"""
        print(f"🔍 Extracting advanced temporal reasoning features from {len(df)} texts...")
        print(f"📊 Features to extract: {self.feature_names}")

        df_result = df.copy()

        # Initialize feature columns
        for feature_name in self.feature_names:
            df_result[feature_name] = 0.0

        successful_extractions = 0
        failed_extractions = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting temporal features"):
            try:
                text = row[text_column]
                if pd.isna(text) or len(str(text).strip()) < 10:
                    failed_extractions += 1
                    continue

                text_str = str(text)
                features = self.extract_temporal_features(text_str)

                # Update dataframe
                for feature_name, value in features.items():
                    if np.isfinite(value):
                        df_result.at[idx, feature_name] = value

                if any(value > 0 for value in features.values()):
                    successful_extractions += 1
                else:
                    failed_extractions += 1

            except Exception as e:
                failed_extractions += 1
                continue

        print(f"✅ Extraction complete: {successful_extractions}/{len(df)} successful")

        if failed_extractions > 0:
            print(f"⚠️ Failed extractions: {failed_extractions}")

        # Feature statistics
        print(f"\n📊 Temporal Feature Statistics:")
        for feature_name in self.feature_names:
            values = df_result[feature_name]
            non_zero = (values > 0).sum()
            mean_val = values.mean()
            std_val = values.std()
            print(f"{feature_name}: μ={mean_val:.4f}, σ={std_val:.4f}, non-zero={non_zero}/{len(df)} ({100*non_zero/len(df):.1f}%)")

        return df_result

def initialize_temporal_extractor(timeml_annotator):
    """
    Initialize the advanced temporal feature extractor with improved granularity

    Parameters:
    -----------
    timeml_annotator : TimeMLAnnotator
        Your existing TimeML annotator instance

    Returns:
    --------
    AdvancedTemporalFeatureExtractor : Initialized extractor with enhanced granularity
    """
    extractor = AdvancedTemporalFeatureExtractor(timeml_annotator)

    print("✅ Enhanced granularity detection integrated automatically!")
    print(f"📊 Expected coverage improvement: 20% → 70%+ for granularity diversity")

    return extractor