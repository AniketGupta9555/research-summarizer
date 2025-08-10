from rouge_score import rouge_scorer

def evaluate_summary(generated_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

if __name__ == "__main__":
    # Example usage
    reference = """This qualitative study investigates the impact of the Japanese comic series ‘Doraemon’
on the maturity process of Vietnamese teenagers. Employing semi-structured interviews
with nine Vietnamese adolescents, the research explores how this popular media
influences cognitive, moral, social, and cultural development. The thematic analysis
reveals four key findings: the enhancement of problem-solving skills and creativity, the
development of moral reasoning and ethical decision-making, the shaping of social
relationships and emotional intelligence, and the influence on cultural identity and
global perspective. The study finds that ‘Doraemon’ inspires creative thinking and
innovative problem-solving, resonating with developmental psychology theories on
cognitive growth during adolescence. It also plays a crucial role in moral development,
presenting complex ethical dilemmas that encourage reflective thinking. Furthermore,
‘Doraemon’ positively impacts social and emotional skills, aiding in the development of
empathy and effective management of social relationships. Lastly, it fosters a broader
understanding of cultural diversity, influencing the participants’ global outlook and
cultural identity. This research contributes to the understanding of how specific media
content can influence adolescent development in diverse cultural contexts. It highlights
the potential of comics as tools for cognitive and moral development, social and
emotional learning, and in fostering global cultural awareness. The findings have
implications for educators, media producers, and policymakers in considering the role
of media in adolescent development."""
    generated = """The study looked at the role media shaping young people drinking cultures and practices related to identity making. The authors also looked at how comics can be used to teach language literacy to children. The study was published in the journal ijaedu international e-journal advances education (ijaedu) The authors hope to use their findings to improve the education of young people across the world through the use of comic books.

[Key Terms]: doraemon, cultural, media, vietnamese, development, study, teenagers, moral, participants, social, studies, influence, global, impact, emotional"""
    
    results = evaluate_summary(generated, reference)
    print("\n=== ROUGE Scores ===")
    for metric, score in results.items():
        print(f"{metric}: Precision={score.precision:.3f}, Recall={score.recall:.3f}, F1={score.fmeasure:.3f}")
