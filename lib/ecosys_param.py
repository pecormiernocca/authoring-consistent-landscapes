import numpy as np

#facteur climatiques
offset_temperature  = 0 # en °C
offset_ground_humidity = 0 # log du facteur multiplicatif: si 3x plus humide que l'actuel, offset = log(3) = 0.47

# calibration des  facteurs climatiques:
# conversion d'une unité physique vers une echelle de 1 à 9

# tempreature (°C)
calibration_temperature = (-15, 25)  # -15°C --> 1 ; 30°C --> 9
calibration_ground_humidity = (-8, 2)   # l'humidité suit une echelle logarithique approximativement entre -12 et 4
calibration_illum = (.01, .25) # illumination entre 0 et 1 (1: cas hypothique de  100% d'illumiation pendant 24h)

# climatic conditions. 0 is december
monthly_avg_temp = np.array([7.7, 7.3, 7.5, 10.5, 13.5, 17.2, 21.6, 24.2, 24.1, 20.1, 16.5, 11.0])
monthly_precip = np.array([20.4, 59.0, 59.1, 86.3, 9.2, 15.8, 41.8, 2.4, 8.6, 24.0, 63.3, 1.6])

# geologie: couleurs sur l'image
geologie_colors = ("ccccfd", "0000ff", "52d252", "adadad", "ffffff" )
# Calcaire, cours d'eau, marne, eboulis, alluvions

##########################
# parametres des plantes #
##########################

# illum = illumination, tmp = temperature, moisture = ground humidity
# parameters = (min, max, compet). min et max between 1 to 9 (from tela botanica)
# compet : consumed resources for 100% density.
# geology: viability (between 0 and 1) for each geological color (defined above)
plants = ([
    {'name': 'Cupressaceae', 'illum': (7, 9,  2), 'temp': (2, 9, 0), 'moisture': (2, 4,  2), 'geology': (1,   0, 1, 1, 1)}, # arbuste, genévrier
    {'name': 'Pinus',        'illum': (7, 9,  5), 'temp': (6, 9, 0), 'moisture': (2, 5,  3), 'geology': (1,   0, 1, 1, 1)}, # Arbre, pin
    {'name': 'Pistacia',     'illum': (7, 9,  2), 'temp': (7, 9, 0), 'moisture': (3, 4,  2), 'geology': (1,   0, 1, 1, 1)}, # Arbuste, pistachiers
    {'name': 'Alnus',        'illum': (7, 9,  5), 'temp': (2, 8, 0), 'moisture': (4, 8,  5), 'geology': (0, 0.8, 0, 0, 1)}, # arbre, Aulne,peuplier, saule
    {'name': 'Apiaceae',     'illum': (3, 9, .2), 'temp': (1, 9, 0), 'moisture': (2, 6, .2), 'geology': (1,   0, 1, 1, 1)}, # plante, Anis, Persil, ...
    {'name': 'Cichorium',    'illum': (7, 9, .2), 'temp': (6, 9, 0), 'moisture': (4, 6, .2), 'geology': (1,   0, 1, 1, 1)}, # plante, Chicorées
    {'name': 'Poaceae',      'illum': (5, 9, .2), 'temp': (1, 9, 0), 'moisture': (1, 8, .2), 'geology': (1,   0, 1, 1, 1)}, # plante, graminées
    {'name': 'Asteraceae',   'illum': (4, 9, .2), 'temp': (3, 9, 0), 'moisture': (1, 7, .2), 'geology': (1,   0, 1, 1, 1)}, # plante, astérasées
    {'name': 'Quercus',      'illum': (7, 9,  5), 'temp': (5, 8, 0), 'moisture': (3, 6,  5), 'geology': (1,   0, 1, 1, 1)}, # arbre, chêne
    {'name': 'Artemisia',    'illum': (7, 9, .2), 'temp': (2, 5, 0), 'moisture': (2, 8, .2), 'geology': (1,   0, 1, 1, 1)}, # plante, artemisia
    {'name': 'Betula',       'illum': (7, 9,  3), 'temp': (3, 5, 0), 'moisture': (4, 7,  3), 'geology': (1,   0, 1, 1, 1)}, # arbre, bouleau
    {'name': 'Carpinus',     'illum': (6, 7,  5), 'temp': (6, 7, 0), 'moisture': (4, 5,  4), 'geology': (1,   0, 1, 1, 1)}, # arbre, charme
    {'name': 'Corylus',      'illum': (5, 6,  4), 'temp': (5, 6, 0), 'moisture': (4, 5,  4), 'geology': (1,   0, 1, 1, 1)}, # arbre, corylus
    {'name': 'Plantago',     'illum': (5, 9, .2), 'temp': (1, 9, 0), 'moisture': (2, 7, .2), 'geology': (1,   0, 1, 1, 1)}, # plante, plantain
    {'name': 'Pinus mugo',   'illum': (7, 9,  3), 'temp': (2, 4, 0), 'moisture': (3, 7,  3), 'geology': (1,   0, 1, 1, 1)}, # arbre, pin montagne
    {'name': 'Quercus ilex', 'illum': (7, 9,  4), 'temp': (7, 9, 0), 'moisture': (4, 7,  4), 'geology': (1,   0, 1, 1, 1)}, # arbre, chêne vert
    {'name': 'Rubiaceae',    'illum': (3, 9, .2), 'temp': (2, 9, 0), 'moisture': (2, 6, .2), 'geology': (1,   0, 1, 1, 1)}, # plante, rubiacées
    {'name': 'Fabaceae',     'illum': (6, 9, .2), 'temp': (2, 9, 0), 'moisture': (2, 6, .2), 'geology': (1,   0, 1, 1, 1)}, # plante, fabacées
])
