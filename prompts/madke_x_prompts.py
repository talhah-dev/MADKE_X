"""MADKE-X prompt additions (Section 4.3 / 4.4).

* ``summary_prompt_cite``  -- summariser prompt used when ``--cite`` is enabled.
  The final answer stays in ``[answer]`` brackets; supporting passages are cited
  with curly-brace identifiers ``{E#}`` (``#`` = 0-based pool index as listed),
  so citations can never collide with the ``[answer]`` bracket parser.
* ``verify_prompt``        -- the bounded verification call (Algorithm 4): answer
  ONLY if a listed passage states the answer, otherwise abstain with
  ``[Unsupported] {E-none}``.

Curly braces that must survive ``str.format`` are doubled (``{{E#}}``).
The example evidence is written on a single ``Evidence:`` line on purpose so
that it is never mistaken for a citable passage of the current question.
"""

summary_prompt_cite = """Please summarize the final answer from answer of all agents. Place the final answer of question in the form of [answer]. After the answer, cite the passage(s) that state the answer using their identifiers in curly braces, e.g. {{E2}} {{E5}}. Only cite passages that are listed under "Evidence (cite with {{E#}})". If no listed passage states the answer, write {{E-none}} instead of a citation.

Here is an example:
Question: Ezzard Charles was a world champion in which sport?
Evidence: {{E0}} Ezzard Mack Charles was an American professional boxer and former world heavyweight champion. {{E3}} Charles was born in Lawrenceville, Georgia, and grew up in Cincinnati, Ohio.
Agent 0: Ezzard Charles was a world champion in the sport of boxing. Therefore, the answer is "boxing".
Agent 1: Ezzard Charles was a world champion in boxing. Therefore, the answer is "boxing".
Answer: Let's think step by step! Both agents state that Ezzard Charles was a boxing champion, and passage {{E0}} states that he was a world heavyweight boxing champion. Therefore, the final answer is [boxing]. {{E0}}
(END OF EXAMPLE)

Question: {question}
{evidence}
{all_answers_from_agents}
Answer: Let's think step by step!
"""

verify_prompt = """This is a verification step. The agents' debate is given below together with the ONLY passages you may rely on. Answer ONLY if one of the listed passages explicitly states the answer to the question: place the answer in the form [answer] and cite the passage identifier(s) in curly braces, e.g. {{E2}}. If no listed passage states the answer, you must abstain by writing exactly: [Unsupported] {{E-none}}. Do not use outside knowledge and do not cite passages that are not listed.

Question: {question}
{evidence}
Debate:
{all_answers_from_agents}
Answer: Let's think step by step!
"""

summary_prompt_cite_fever = """Please summarize the final answer from answer of all agents. Place the final answer of question in the form of [answer], where the answer is one of SUPPORTS, REFUTES or NOT ENOUGH INFO. After the answer, cite the passage(s) that justify the label using their identifiers in curly braces, e.g. {{E2}} {{E5}}. Only cite passages that are listed under "Evidence (cite with {{E#}})". If no listed passage justifies the label, write {{E-none}} instead of a citation.

Here is an example:
Question: Robert J. O'Neill was born April 10, 1976.
Evidence: {{E1}} Robert J. O'Neill (born 10 April 1976) is a former United States Navy SEAL, TV news contributor, and author.
Agent 0: Robert J. O'Neill was indeed born on April 10, 1976. Therefore, the answer is [SUPPORTS].
Agent 1: Robert J. O'Neill (born 10 April 1976) is a former United States Navy SEAL. Therefore, the answer is [SUPPORTS].
Answer: Let's think step by step! Both agents agree and passage {{E1}} states the birth date. Therefore, the final answer is [SUPPORTS]. {{E1}}
(END OF EXAMPLE)

Question: {question}
{evidence}
{all_answers_from_agents}
Answer: Let's think step by step!
"""

verify_prompt_fever = """This is a verification step. The agents' debate is given below together with the ONLY passages you may rely on. Return the label SUPPORTS or REFUTES in the form [answer] ONLY if one of the listed passages explicitly justifies it, citing the passage identifier(s) in curly braces, e.g. {{E2}}. If no listed passage justifies a label, you must abstain by writing exactly: [Unsupported] {{E-none}}. Do not use outside knowledge.

Question: {question}
{evidence}
Debate:
{all_answers_from_agents}
Answer: Let's think step by step!
"""
