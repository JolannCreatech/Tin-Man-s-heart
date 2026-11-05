# Musique
avoir les pistes melodies, Accords, Structure harmonique, Tempo, Éventuellement les paroles
l’IA génère des accompagnements qui respectent cette structure.
Je commencerais par une génération complète de l’accompagnement.
Mais je construirais le système de manière modulaire pour pouvoir passer à plusieurs pistes indépendantes ensuite.

# Modele 
Magenta RealTime 2
MIDI musical instrument digital interface


"""
BPM -> features for prompt       If I only had a heart 
            |_________________________|   |
                         |                |
                    Magenta Model         |
                         |________________|
                                  |
                               assembly
                                  |
                              Live music
‘‘‘