import os
import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import numpy as np
from collections import defaultdict

# ----------------------------
# Load all JSON docs from /app/input
# ----------------------------
def load_documents(input_dir):
    documents = []
    for json_file in Path(input_dir).glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["filename"] = json_file.name
            documents.append(data)
    return documents

# ----------------------------
# Load persona and job JSON
# ----------------------------
def load_persona_job(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # fallback
        return {
            "persona": "researcher",
            "job_to_be_done": "analyze documents"
        }

# ----------------------------
# Enhanced section extraction with content simulation
# ----------------------------
def extract_sections_with_content(documents):
    sections = []
    for doc in documents:
        doc_title = doc.get("title", "")
        
        for item in doc.get("outline", []):
            # Simulate content extraction based on section title and context
            section_title = item.get("text", "")
            level = item.get("level", "")
            page = item.get("page", 1)
            
            # Generate simulated content based on section title and document context
            # In a real implementation, this would extract actual text content
            simulated_content = generate_section_content(section_title, doc_title, level)
            
            sections.append({
                "document": doc["filename"],
                "page": page,
                "section_title": section_title,
                "level": level,
                "content": simulated_content,
                "doc_title": doc_title
            })
    return sections

# ----------------------------
# Generate simulated content for sections
# ----------------------------
def generate_section_content(section_title, doc_title, level):
    """
    Simulate content extraction. In real implementation, this would
    extract actual text content from the PDF sections.
    """
    # Create contextual content based on title and level
    base_content = f"This section discusses {section_title.lower()} in the context of {doc_title}. "
    
    if level == "H1":
        base_content += f"This is a major section covering fundamental aspects of {section_title}. "
        base_content += "It provides comprehensive coverage of key concepts, methodologies, and findings. "
    elif level == "H2":
        base_content += f"This subsection delves deeper into specific aspects of {section_title}. "
        base_content += "It includes detailed analysis, examples, and supporting evidence. "
    elif level == "H3":
        base_content += f"This detailed section focuses on particular elements of {section_title}. "
        base_content += "It provides specific information, data points, and technical details. "
    
    # Add some domain-specific keywords based on common section titles
    keywords = extract_domain_keywords(section_title)
    if keywords:
        base_content += f"Key topics include: {', '.join(keywords)}. "
    
    return base_content

# ----------------------------
# Extract domain keywords from section titles
# ----------------------------
def extract_domain_keywords(section_title):
    """Extract relevant keywords based on section title patterns"""
    title_lower = section_title.lower()
    keywords = []
    
    # Academic/Research keywords
    if any(word in title_lower for word in ['method', 'approach', 'algorithm']):
        keywords.extend(['methodology', 'implementation', 'evaluation'])
    
    if any(word in title_lower for word in ['result', 'finding', 'analysis']):
        keywords.extend(['data', 'performance', 'metrics', 'comparison'])
    
    if any(word in title_lower for word in ['introduction', 'background']):
        keywords.extend(['context', 'motivation', 'objectives'])
    
    if any(word in title_lower for word in ['conclusion', 'discussion']):
        keywords.extend(['implications', 'limitations', 'future work'])
    
    # Business keywords
    if any(word in title_lower for word in ['revenue', 'financial', 'market']):
        keywords.extend(['growth', 'trends', 'performance', 'strategy'])
    
    # Technical keywords
    if any(word in title_lower for word in ['system', 'architecture', 'design']):
        keywords.extend(['implementation', 'components', 'integration'])
    
    return keywords[:5]  # Limit to top 5 keywords

# ----------------------------
# Enhanced persona-aware ranking
# ----------------------------
def rank_sections_persona_aware(sections, persona, job):
    """
    Enhanced ranking that considers both persona expertise and job requirements
    """
    # Create persona-enhanced query
    persona_keywords = get_persona_keywords(persona)
    job_keywords = extract_job_keywords(job)
    
    enhanced_query = f"{persona} {job} {' '.join(persona_keywords)} {' '.join(job_keywords)}"
    
    # Prepare texts for vectorization
    section_texts = []
    for s in sections:
        # Combine title and content for better context
        combined_text = f"{s['section_title']} {s['content']}"
        section_texts.append(combined_text)
    
    all_texts = section_texts + [enhanced_query]
    
    # Use TF-IDF with enhanced parameters
    vectorizer = TfidfVectorizer(
        stop_words="english", 
        max_features=2000,
        ngram_range=(1, 2),  # Include bigrams
        min_df=1,
        max_df=0.95
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        query_vector = tfidf_matrix[-1]
        section_vectors = tfidf_matrix[:-1]
        
        # Calculate cosine similarity
        scores = cosine_similarity(query_vector, section_vectors).flatten()
        
        # Apply persona-specific boosting
        boosted_scores = apply_persona_boosting(sections, scores, persona, job)
        
        # Add scores to sections
        for i, s in enumerate(sections):
            s["similarity_score"] = float(boosted_scores[i])
            s["base_score"] = float(scores[i])
        
    except Exception as e:
        print(f"Warning: TF-IDF failed, using fallback scoring: {e}")
        # Fallback to simple keyword matching
        for i, s in enumerate(sections):
            s["similarity_score"] = calculate_keyword_score(s, persona, job)
            s["base_score"] = s["similarity_score"]
    
    # Sort by score
    ranked = sorted(sections, key=lambda x: x["similarity_score"], reverse=True)
    
    # Add rank
    for idx, s in enumerate(ranked, start=1):
        s["importance_rank"] = idx
    
    return ranked

# ----------------------------
# Get persona-specific keywords
# ----------------------------
def get_persona_keywords(persona):
    persona_lower = persona.lower()
    keywords = []
    
    if 'researcher' in persona_lower or 'phd' in persona_lower:
        keywords = ['methodology', 'analysis', 'findings', 'research', 'study', 'experimental', 'theoretical']
    elif 'student' in persona_lower:
        keywords = ['concepts', 'fundamentals', 'examples', 'learning', 'understanding', 'basics']
    elif 'analyst' in persona_lower or 'investment' in persona_lower:
        keywords = ['trends', 'performance', 'metrics', 'data', 'financial', 'analysis', 'comparison']
    elif 'engineer' in persona_lower:
        keywords = ['implementation', 'technical', 'system', 'design', 'architecture', 'solution']
    elif 'manager' in persona_lower or 'business' in persona_lower:
        keywords = ['strategy', 'operations', 'management', 'process', 'efficiency', 'results']
    else:
        keywords = ['information', 'details', 'overview', 'summary', 'key points']
    
    return keywords

# ----------------------------
# Extract job-specific keywords
# ----------------------------
def extract_job_keywords(job):
    job_lower = job.lower()
    keywords = []
    
    if 'literature review' in job_lower:
        keywords = ['methodology', 'findings', 'comparison', 'analysis', 'research']
    elif 'exam preparation' in job_lower or 'study' in job_lower:
        keywords = ['concepts', 'key points', 'fundamentals', 'examples', 'practice']
    elif 'financial' in job_lower or 'revenue' in job_lower:
        keywords = ['financial', 'revenue', 'profit', 'trends', 'performance', 'metrics']
    elif 'summary' in job_lower or 'summarize' in job_lower:
        keywords = ['overview', 'key points', 'main findings', 'conclusions', 'results']
    elif 'analysis' in job_lower or 'analyze' in job_lower:
        keywords = ['data', 'trends', 'patterns', 'comparison', 'evaluation', 'metrics']
    
    return keywords

# ----------------------------
# Apply persona-specific boosting
# ----------------------------
def apply_persona_boosting(sections, scores, persona, job):
    boosted_scores = scores.copy()
    
    for i, section in enumerate(sections):
        boost_factor = 1.0
        section_title = section['section_title'].lower()
        content = section['content'].lower()
        
        # Persona-specific boosts
        if 'researcher' in persona.lower():
            if any(word in section_title for word in ['method', 'result', 'analysis', 'finding']):
                boost_factor += 0.3
            if any(word in content for word in ['experimental', 'data', 'statistical']):
                boost_factor += 0.2
        
        elif 'student' in persona.lower():
            if any(word in section_title for word in ['introduction', 'basic', 'fundamental', 'concept']):
                boost_factor += 0.3
            if section['level'] == 'H1':  # Main sections are important for students
                boost_factor += 0.2
        
        elif 'analyst' in persona.lower():
            if any(word in section_title for word in ['trend', 'performance', 'financial', 'metric']):
                boost_factor += 0.3
            if any(word in content for word in ['revenue', 'growth', 'market']):
                boost_factor += 0.2
        
        # Job-specific boosts
        if 'literature review' in job.lower():
            if any(word in section_title for word in ['method', 'approach', 'result', 'conclusion']):
                boost_factor += 0.2
        
        elif 'exam preparation' in job.lower():
            if any(word in section_title for word in ['concept', 'principle', 'example', 'problem']):
                boost_factor += 0.2
        
        boosted_scores[i] *= boost_factor
    
    return boosted_scores

# ----------------------------
# Fallback keyword scoring
# ----------------------------
def calculate_keyword_score(section, persona, job):
    """Fallback scoring method using simple keyword matching"""
    score = 0.0
    
    text = f"{section['section_title']} {section['content']}".lower()
    persona_keywords = get_persona_keywords(persona)
    job_keywords = extract_job_keywords(job)
    
    # Score based on keyword presence
    for keyword in persona_keywords:
        if keyword in text:
            score += 0.1
    
    for keyword in job_keywords:
        if keyword in text:
            score += 0.15
    
    # Boost based on section level
    if section['level'] == 'H1':
        score += 0.05
    elif section['level'] == 'H2':
        score += 0.03
    
    return score

# ----------------------------
# Extract and rank sub-sections
# ----------------------------
def extract_subsections(top_sections, persona, job):
    """
    Extract sub-sections from top-ranked sections and rank them
    """
    subsections = []
    
    for section in top_sections:
        # Generate sub-sections based on the main section
        # In real implementation, this would extract actual sub-sections from the document
        generated_subs = generate_subsections(section, persona, job)
        subsections.extend(generated_subs)
    
    # Rank sub-sections
    subsection_texts = [sub['refined_text'] for sub in subsections]
    
    if subsection_texts:
        persona_keywords = get_persona_keywords(persona)
        job_keywords = extract_job_keywords(job)
        query = f"{persona} {job} {' '.join(persona_keywords)} {' '.join(job_keywords)}"
        
        all_texts = subsection_texts + [query]
        
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[-1]
            sub_vectors = tfidf_matrix[:-1]
            
            scores = cosine_similarity(query_vector, sub_vectors).flatten()
            
            for i, sub in enumerate(subsections):
                sub["relevance_score"] = float(scores[i])
        
        except:
            # Fallback scoring
            for sub in subsections:
                sub["relevance_score"] = calculate_keyword_score({
                    'section_title': '',
                    'content': sub['refined_text']
                }, persona, job)
    
    # Sort and rank
    ranked_subs = sorted(subsections, key=lambda x: x["relevance_score"], reverse=True)
    
    for idx, sub in enumerate(ranked_subs, start=1):
        sub["importance_rank"] = idx
    
    return ranked_subs[:20]  # Return top 20 sub-sections

# ----------------------------
# Generate sub-sections from main sections
# ----------------------------
def generate_subsections(section, persona, job):
    """
    Generate sub-sections based on the main section content
    In real implementation, this would parse actual document sub-sections
    """
    subsections = []
    base_content = section['content']
    
    # Generate 2-4 sub-sections per main section
    sub_topics = generate_sub_topics(section['section_title'], persona, job)
    
    for i, sub_topic in enumerate(sub_topics):
        refined_text = generate_refined_subsection_text(
            sub_topic, base_content, section['section_title'], persona, job
        )
        
        subsections.append({
            "document": section['document'],
            "page": section['page'],
            "refined_text": refined_text,
            "parent_section": section['section_title'],
            "sub_topic": sub_topic
        })
    
    return subsections

# ----------------------------
# Generate sub-topics
# ----------------------------
def generate_sub_topics(section_title, persona, job):
    """Generate relevant sub-topics based on section title and context"""
    title_lower = section_title.lower()
    sub_topics = []
    
    if 'method' in title_lower or 'approach' in title_lower:
        sub_topics = ['Methodology Overview', 'Implementation Details', 'Evaluation Criteria', 'Comparative Analysis']
    elif 'result' in title_lower or 'finding' in title_lower:
        sub_topics = ['Key Findings', 'Statistical Analysis', 'Performance Metrics', 'Interpretation']
    elif 'introduction' in title_lower:
        sub_topics = ['Background Context', 'Problem Statement', 'Objectives', 'Scope and Limitations']
    elif 'conclusion' in title_lower:
        sub_topics = ['Summary of Findings', 'Implications', 'Future Directions', 'Recommendations']
    else:
        # Generic sub-topics
        sub_topics = ['Core Concepts', 'Detailed Analysis', 'Practical Applications', 'Key Insights']
    
    return sub_topics[:3]  # Return top 3 sub-topics

# ----------------------------
# Generate refined subsection text
# ----------------------------
def generate_refined_subsection_text(sub_topic, base_content, section_title, persona, job):
    """
    Generate refined text for sub-sections based on persona and job requirements
    """
    # Create persona-tailored content
    refined_text = f"[{sub_topic}] "
    
    if 'researcher' in persona.lower():
        refined_text += f"From a research perspective, this aspect of {section_title} involves detailed examination of {sub_topic.lower()}. "
        refined_text += "This includes methodological considerations, empirical evidence, and analytical frameworks. "
    elif 'student' in persona.lower():
        refined_text += f"For learning purposes, {sub_topic.lower()} in {section_title} covers fundamental concepts that are essential to understand. "
        refined_text += "This includes key definitions, examples, and practical applications. "
    elif 'analyst' in persona.lower():
        refined_text += f"From an analytical standpoint, {sub_topic.lower()} within {section_title} provides crucial insights for decision-making. "
        refined_text += "This includes data trends, performance indicators, and strategic implications. "
    else:
        refined_text += f"This section on {sub_topic.lower()} provides important information related to {section_title}. "
    
    # Add job-specific context
    if 'literature review' in job.lower():
        refined_text += "This content is particularly relevant for systematic literature analysis and comparison with existing research. "
    elif 'exam preparation' in job.lower():
        refined_text += "This material is important for examination purposes and should be studied thoroughly. "
    elif 'financial analysis' in job.lower():
        refined_text += "This information is crucial for financial evaluation and investment decision-making. "
    
    # Add some of the base content context
    refined_text += f"Building on the foundation of {section_title}, this refined analysis provides targeted insights. "
    
    return refined_text

# ----------------------------
# Build enhanced output JSON
# ----------------------------
def build_enhanced_output(ranked_sections, ranked_subsections, persona, job, documents):
    """Build output matching the required format"""
    
    # Get top 10 sections
    top_sections = ranked_sections[:10]
    
    output = {
        "metadata": {
            "persona": persona,
            "job_to_be_done": job,
            "documents": [doc for doc in documents],
            "processing_timestamp": datetime.now().isoformat()
        },
        "sections": [],
        "sub_sections": []
    }
    
    # Add sections
    for s in top_sections:
        output["sections"].append({
            "document": s["document"],
            "page": s["page"],
            "section_title": s["section_title"],
            "importance_rank": s["importance_rank"]
        })
    
    # Add sub-sections (top 15)
    for sub in ranked_subsections[:15]:
        output["sub_sections"].append({
            "document": sub["document"],
            "page": sub["page"],
            "refined_text": sub["refined_text"],
            "importance_rank": sub["importance_rank"]
        })
    
    return output

# ----------------------------
# Main function
# ----------------------------
def main():
    input_dir = "/app/input"
    output_dir = "/app/output"
    persona_job_path = "/app/test_persona_job.json"

    print("🚀 Starting persona-driven document intelligence...")
    
    # Load data
    documents = load_documents(input_dir)
    if not documents:
        print("⚠️ No JSON documents found in input folder.")
        return
    
    print(f"📚 Loaded {len(documents)} documents")
    
    persona_job = load_persona_job(persona_job_path)
    persona = persona_job.get("persona", "")
    job = persona_job.get("job_to_be_done", "")
    
    print(f"👤 Persona: {persona}")
    print(f"🎯 Job: {job}")

    # Extract sections with enhanced content
    print("📖 Extracting sections with content...")
    sections = extract_sections_with_content(documents)
    print(f"📄 Extracted {len(sections)} sections")

    # Rank sections with persona awareness
    print("🧠 Ranking sections with persona-aware algorithm...")
    ranked_sections = rank_sections_persona_aware(sections, persona, job)
    
    # Extract and rank sub-sections
    print("🔍 Extracting and ranking sub-sections...")
    top_sections_for_subs = ranked_sections[:5]  # Use top 5 sections for sub-section extraction
    ranked_subsections = extract_subsections(top_sections_for_subs, persona, job)
    print(f"📝 Generated {len(ranked_subsections)} sub-sections")

    # Build enhanced output
    doc_names = [doc["filename"] for doc in documents]
    output_data = build_enhanced_output(ranked_sections, ranked_subsections, persona, job, doc_names)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("✅ Processing complete! Enhanced output saved to output/output.json")
    print(f"📊 Top sections: {len(output_data['sections'])}")
    print(f"📋 Sub-sections: {len(output_data['sub_sections'])}")

if __name__ == "__main__":
    main()