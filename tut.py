# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:07:27 2023

@author: rkpal
"""

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

s1 = "Sun, an average star"
s2 = 'The solar system is huge.'
s3 = "best fries is still McDonald"


s1 = 'patient with fever, temperature 40 degrees'
s2 = 'patient with pyrexia'
s3 = 'mrsa positive patient, to be isolated'


# Q: What behavioural change is needed for reducing obesity?

# PubMed_abstracts_hsor
s1 = '''Multiple studies suggest that obesity reduction strategies should focus on a combination of dietary and behavioural interventions. Nutritional strategies that 
reduce emotional eating, late eating, and grazing can help manage obesity. Interventions targeting susceptibility to hunger, food cravings, and emotional dysregulation 
may also be beneficial. Cognitive remediation-enabled cognitive behaviour therapy (CR-CBT) can improve cognitive flexibility and response inhibition, aiding in reducing 
binge eating and promoting weight loss. Additionally, addressing restrictive eating behaviour disorders, increasing knowledge about eating behaviours, and managing 
appetite in infancy and early childhood can contribute to obesity reduction. Reducing energy intake, particularly from soft drinks, and increasing self-efficacy for 
weight control are also suggested. Physical activity plays a crucial role in obesity management. Studies suggest that reducing sedentary behaviour, increasing physical
 activity, and integrating motivational interviewing and cognitive behaviour therapy can lead to improvements in physical activity and body composition. 
Smoking cessation is also associated with obesity reduction. Healthcare professionals' behaviour and care organization, including educational interventions for 
general practitioners and compliance with obesity guidelines, can help manage obesity. Parents' responsive parenting, recognizing hunger-satiety cues, and 
responding appropriately to distress may impact childhood overweight and obesity. Strategies for managing emotions, promoting positive mood, reducing impulsivity, 
and adopting health-promoting behaviour are also necessary. A combination of healthy lifestyle behaviours such as maintaining a balanced diet, engaging in regular 
physical activity, not smoking, limiting alcohol intake, and obtaining higher education may help reduce the risk of obesity. 
Interventions should also target the development of healthy eating habits and physical activity, control desires and rewards, emotional control, and increase
 knowledge about healthy eating. Public health programmes addressing obesity in individuals with a low socioeconomic status, promoting healthier eating habits,
 and increasing physical activity levels are needed. Parents need to be educated and supported in making healthy lifestyle choices for their infants. 
Collaborative prevention and treatment strategies should address the shared risk factors for obesity and eating disorders, promoting healthy eating habits and regular
 physical activity. Family therapy may be an effective approach in preventing severe obesity during puberty. 
Finally, reducing sedentary behaviours, such as watching television and using computers, and improving diet are essential for obesity reduction. '''

# SciSpace
s2 = '''Behavioral change is crucial in reducing obesity. Various behavior modification techniques have been identified, including goal-setting, self-monitoring, 
stimulus control, cognitive restructuring, stress management, problem-solving, and social support and information. These techniques have been found to be effective 
in achieving and maintaining weight loss, particularly in long-term weight maintenance [1]. Interventions targeting youth with low socioeconomic status (SES) have 
shown promising results in preventing or reducing obesity. Studies have found that interventions with six or more behavioral change techniques (BCTs) and specific BCTs 
such as problem-solving, social support, instruction on how to perform the behavior, identification of self as a role model, and demonstration of the behavior have
 larger effects in decreasing BMI among youth with low SES [2]. Parental involvement, goal setting, motivational counseling, role modeling, and restructuring the 
 physical environment have been associated with positive dietary and physical activity behavior change in children, contributing to the reduction of childhood
 obesity [3]. Theory-based lifestyle interventions, incorporating behavior change techniques such as behavior goal setting, self-monitoring, instructions on behavior, 
 and credible sources, have shown promise in the treatment of overweight/obesity in breast cancer survivors [4]. Advances in behavior change science theories and methods 
 have provided insights into the factors that predict the adoption and maintenance of health behaviors, and how these factors can be implemented in effective behavior 
 change interventions for obesity management [5].'''


# Perplexity
s3 = '''To reduce obesity, effective behavioral changes are crucial. Evidence from PubMed suggests several strategies:
Key Behavioral Changes for Reducing Obesity:
Self-Regulation of Eating Behaviors: Implementing strategies like goal setting, action plans, self-monitoring, and relapse prevention plans are important for weight loss maintenance
1. Utilizing Behavior Change Techniques (BCTs): Effective BCTs include providing information on the consequences of behavior, providing rewards contingent on successful behavior, and facilitating social comparison
Incorporating Technology: The use of technology, such as mobile apps and web-based interventions, can enhance engagement and convenience in behavior change programs for obesity management
Theory-Based Interventions: Incorporating formal theories and behavior change techniques informed by theory can improve the effectiveness of health behavior change interventions for obesity management
Combining Effective BCTs: Designing programs that combine effective BCTs identified through research can enhance the impact of interventions aimed at preventing and managing obesity in both children and adults
By focusing on these evidence-based behavioral strategies, individuals can make sustainable changes to combat obesity effectively.'''




# map sentence to vector (string to numbers)
embed_s1 = model.encode(s1)
embed_s2 = model.encode(s2)
embed_s3 = model.encode(s3)

sim1 = util.cos_sim(embed_s1,embed_s2) # similar
sim2 = util.cos_sim(embed_s1,embed_s3) # not related
sim3 = util.cos_sim(embed_s2,embed_s3)

print(sim1) # gives 0.487 (reasonable similarity)
print(sim2) # gives 0.07 (lowest is 0)
print(sim3)

"""
z = zip(sentences, embeddings)

for s, e in z:
    print("Sentence", s)
    print("Embedding", e)
    print("")
    
"""