"""Build the 3-layer task prompt that drives browser-use agents.

Layer 1: Site Knowledge — static description of the Travliaq UI
Layer 2: Persona Instructions — generated from the persona JSON
Layer 3: Conversation Flow — ordered phases with goals and widget hints
"""

from .persona_loader import PersonaDefinition

# ---------------------------------------------------------------------------
# Layer 1: Site Knowledge (static)
# ---------------------------------------------------------------------------

SITE_KNOWLEDGE_FR = """
=== CONNAISSANCE DU SITE ===

Tu vas interagir avec travliaq.com, un site de planification de voyage avec un chatbot IA.

NAVIGATION:
- L'URL de départ est : {planner_url}
- Le panneau gauche (35%) contient le CHAT. Le panneau droit (65%) contient une CARTE.
- Concentre-toi UNIQUEMENT sur le panneau de chat. Ignore la carte.

INTERFACE DE CHAT:
- Le champ de texte pour écrire a l'aria-label "Envoyer un message..." (FR) ou "Send a message..." (EN).
- Le bouton d'envoi a l'aria-label "Envoyer" (FR) ou "Send" (EN).
- Au-dessus du champ de texte, il y a des "Smart Suggestions" : des boutons/chips cliquables (role="toolbar").
- Quand l'assistant réfléchit, un indicateur de frappe (3 points animés) apparaît. ATTENDS qu'il disparaisse avant d'agir.
- Si un message d'erreur apparaît, clique sur le bouton "Réessayer" / "Retry".

WIDGETS INTERACTIFS (apparaissent dans les messages de l'assistant):
- CALENDRIER (datePicker / dateRangePicker) : Une grille de jours. Clique sur les dates souhaitées, puis clique "Confirmer".
- SÉLECTEUR DE VOYAGEURS (travelersSelector) : Des boutons +/- pour Adultes, Enfants, Bébés. Ajuste les compteurs puis clique "Confirmer".
- TYPE DE VOYAGE (tripTypeConfirm) : 3 boutons — "Aller-retour", "Aller simple", "Multi-destinations". Clique celui qui convient.
- STYLE DE PRÉFÉRENCES (preferenceStyle) : Des curseurs/sliders à déplacer. Clique "Continuer" quand c'est fait.
- INTÉRÊTS (preferenceInterests) : Des tags/chips cliquables. Sélectionne ceux qui t'intéressent, puis "Continuer".
- SUGGESTIONS DE DESTINATIONS (destinationSuggestions) : Une grille de cartes avec photos. Clique sur celle qui te plaît.
- BUDGET (budgetRangeSlider) : Un slider de plage min-max. Ajuste puis "Confirmer".
- SÉLECTION DE VILLE (citySelector) : Une liste de villes. Clique sur ta ville.
- CONFIRMATION D'AÉROPORT (airportConfirmation) : Des boutons avec les aéroports. Clique pour confirmer.
- Après confirmation, le widget se replie avec un bouton "Modifier" si tu veux changer.

LIEN D'ENVOI DE LOGS (en bas du chat):
- Tout en bas du chat, sous le champ de texte, il y a un lien "Cliquez ici pour nous aider".
- Ce lien apparaît après 3 messages envoyés.
- À LA FIN de ta conversation (dernière action), tu DOIS cliquer sur ce lien.
- Une popup va s'ouvrir. Écris un court résumé de ton expérience dans le champ de texte de la popup, puis envoie-le.

ERREURS DE CARTE (IGNORER):
- Le panneau droit (carte) peut afficher une erreur WebGL ou être complètement vide. C'est NORMAL.
- IGNORE complètement la carte et toute erreur visuelle sur le panneau droit.
- NE CLIQUE JAMAIS sur la carte, ses messages d'erreur, ou ses boutons de rechargement.
- Concentre-toi UNIQUEMENT sur le panneau de chat à GAUCHE.

RÈGLES IMPORTANTES:
- Quand un widget apparaît, INTERAGIS AVEC LUI en cliquant sur ses éléments. Ne tape PAS par-dessus.
- Attends toujours que l'indicateur de frappe disparaisse avant d'envoyer ton prochain message.
- Lis la réponse complète de l'assistant avant d'agir.
- Si l'assistant pose une question, RÉPONDS-Y en restant dans ton personnage.
"""

SITE_KNOWLEDGE_EN = """
=== SITE KNOWLEDGE ===

You will interact with travliaq.com, a travel planning website with an AI chatbot.

NAVIGATION:
- Starting URL: {planner_url}
- The left panel (35%) contains the CHAT. The right panel (65%) contains a MAP.
- Focus ONLY on the chat panel. Ignore the map.

CHAT INTERFACE:
- The text input has aria-label "Send a message..." (EN) or "Envoyer un message..." (FR).
- The send button has aria-label "Send" (EN) or "Envoyer" (FR).
- Above the input, there are "Smart Suggestions": clickable chip buttons (role="toolbar").
- When the assistant is thinking, a typing indicator (3 animated dots) appears. WAIT for it to disappear before acting.
- If an error message appears, click the "Retry" / "Réessayer" button.

INTERACTIVE WIDGETS (appear within assistant messages):
- CALENDAR (datePicker / dateRangePicker): A grid of days. Click the desired dates, then click "Confirm" / "Confirmer".
- TRAVELER SELECTOR (travelersSelector): +/- buttons for Adults, Children, Infants. Adjust counts then click "Confirm".
- TRIP TYPE (tripTypeConfirm): 3 buttons — "Round-trip" / "Aller-retour", "One-way" / "Aller simple", "Multi-city" / "Multi-destinations". Click the right one.
- PREFERENCE STYLE (preferenceStyle): Sliders to drag. Click "Continue" / "Continuer" when done.
- INTERESTS (preferenceInterests): Clickable tag chips. Select your interests, then "Continue".
- DESTINATION SUGGESTIONS (destinationSuggestions): A grid of photo cards. Click the one you like.
- BUDGET (budgetRangeSlider): A min-max range slider. Adjust then "Confirm".
- CITY SELECTOR (citySelector): A list of cities. Click your city.
- AIRPORT CONFIRMATION (airportConfirmation): Airport buttons. Click to confirm.
- After confirming, widgets collapse with a "Modify" / "Modifier" button.

LOG SUBMISSION LINK (bottom of chat):
- At the very bottom of the chat, below the text input, there is a link "Cliquez ici pour nous aider".
- This link appears after 3 messages have been sent.
- As your FINAL action in the conversation, you MUST click this link.
- A popup will open. Write a short summary of your experience in the popup's text field, then submit it.

MAP ERRORS (IGNORE):
- The right panel (map) may display a WebGL error or be completely blank. This is NORMAL.
- IGNORE the map entirely and any visual errors on the right panel.
- NEVER click on the map, its error messages, or its reload buttons.
- Focus ONLY on the chat panel on the LEFT.

IMPORTANT RULES:
- When a widget appears, INTERACT WITH IT by clicking its elements. Do NOT type over it.
- Always wait for the typing indicator to disappear before sending your next message.
- Read the assistant's full response before acting.
- If the assistant asks a question, ANSWER it while staying in character.
"""


# ---------------------------------------------------------------------------
# Layer 2: Persona Instructions (dynamic)
# ---------------------------------------------------------------------------

def _build_persona_section(persona: PersonaDefinition) -> str:
    """Generate persona-specific instructions."""
    lang = "French" if persona.language == "fr" else "English"

    travelers_desc = []
    for key, count in persona.travel_profile.travelers.items():
        if count > 0:
            travelers_desc.append(f"{count} {key}")
    travelers_str = ", ".join(travelers_desc) if travelers_desc else "solo"

    preferred = ", ".join(persona.travel_profile.preferred_destinations) or "pas de préférence"
    avoided = ", ".join(persona.travel_profile.avoided) or "rien en particulier"
    traits = ", ".join(persona.personality_traits) or "neutre"

    style = persona.conversation_style
    verbosity_desc = {
        "low": "concis, phrases courtes" if persona.language == "fr" else "concise, short sentences",
        "medium": "normal, ni trop long ni trop court" if persona.language == "fr" else "normal, balanced length",
        "high": "bavard, détaillé, pose beaucoup de questions" if persona.language == "fr" else "talkative, detailed, asks many questions",
    }
    formality_desc = {
        "formal": "vouvoiement, ton professionnel" if persona.language == "fr" else "formal, professional tone",
        "casual": "tutoiement amical, décontracté" if persona.language == "fr" else "friendly, relaxed tone",
        "mixed": "mélange de formel et informel" if persona.language == "fr" else "mix of formal and informal",
    }

    if persona.language == "fr":
        section = f"""
=== TON PERSONNAGE ===

Tu es {persona.name}, {persona.age or '?'} ans.
{persona.role}

Personnalité : {traits}
Style de communication : {verbosity_desc.get(style.verbosity, style.verbosity)}, {formality_desc.get(style.formality, style.formality)}
{"Tu poses des questions de suivi." if style.asks_questions else "Tu réponds directement sans poser de questions."}
{"Tu peux exprimer de la frustration si l'expérience est lente ou confuse." if style.expresses_frustration else "Tu restes patient(e) même si c'est lent."}
{"Tu peux changer d'avis en cours de conversation." if style.changes_mind else "Tu restes cohérent(e) dans tes choix."}

LANGUE : Parle UNIQUEMENT en {lang}. Toutes tes réponses doivent être en {lang}.

PROFIL DE VOYAGE :
- Groupe : {persona.travel_profile.group_type} — {travelers_str}
- Budget : {persona.travel_profile.budget_range}
- Destinations souhaitées : {preferred}
- À éviter : {avoided}
- Type de voyage : {persona.travel_profile.trip_type}
- Période : {persona.travel_profile.preferred_month or 'flexible'}
- Durée : {persona.travel_profile.trip_duration or 'flexible'}
"""
    else:
        section = f"""
=== YOUR CHARACTER ===

You are {persona.name}, {persona.age or '?'} years old.
{persona.role}

Personality: {traits}
Communication style: {verbosity_desc.get(style.verbosity, style.verbosity)}, {formality_desc.get(style.formality, style.formality)}
{"You ask follow-up questions." if style.asks_questions else "You respond directly without asking questions."}
{"You may express frustration if the experience is slow or confusing." if style.expresses_frustration else "You stay patient even if things are slow."}
{"You may change your mind during the conversation." if style.changes_mind else "You stay consistent in your choices."}

LANGUAGE: Speak ONLY in {lang}. All your responses must be in {lang}.

TRAVEL PROFILE:
- Group: {persona.travel_profile.group_type} — {travelers_str}
- Budget: {persona.travel_profile.budget_range}
- Preferred destinations: {preferred}
- Avoiding: {avoided}
- Trip type: {persona.travel_profile.trip_type}
- When: {persona.travel_profile.preferred_month or 'flexible'}
- Duration: {persona.travel_profile.trip_duration or 'flexible'}
"""
    return section


# ---------------------------------------------------------------------------
# Layer 3: Conversation Flow (dynamic)
# ---------------------------------------------------------------------------

def _build_flow_section(persona: PersonaDefinition) -> str:
    """Generate the ordered conversation flow instructions."""
    if persona.language == "fr":
        header = "\n=== DÉROULEMENT DE LA CONVERSATION ===\n\nSuis ces phases dans l'ordre. Reste dans ton personnage à chaque instant.\n"
    else:
        header = "\n=== CONVERSATION FLOW ===\n\nFollow these phases in order. Stay in character at all times.\n"

    phases = []
    for i, goal in enumerate(persona.conversation_goals, 1):
        phase_block = f"\nPhase {i} — {goal.phase.upper()}:\n"
        phase_block += f"  Objectif: {goal.goal}\n" if persona.language == "fr" else f"  Goal: {goal.goal}\n"

        if goal.example_message:
            if persona.language == "fr":
                phase_block += f'  Message de départ: "{goal.example_message}"\n'
            else:
                phase_block += f'  Starting message: "{goal.example_message}"\n'

        if goal.widget_interactions:
            if persona.language == "fr":
                phase_block += "  Interactions widgets:\n"
            else:
                phase_block += "  Widget interactions:\n"
            for wi in goal.widget_interactions:
                phase_block += f"    - {wi}\n"

        if goal.min_messages:
            if persona.language == "fr":
                phase_block += f"  Envoie au moins {goal.min_messages} messages dans cette phase.\n"
            else:
                phase_block += f"  Send at least {goal.min_messages} messages in this phase.\n"

        if goal.success_indicator:
            if persona.language == "fr":
                phase_block += f"  Indicateur de succès: {goal.success_indicator}\n"
            else:
                phase_block += f"  Success indicator: {goal.success_indicator}\n"

        phases.append(phase_block)

    if persona.language == "fr":
        footer = """
RÈGLES DE RYTHME:
- Attends que l'indicateur de frappe (3 points) disparaisse avant d'envoyer.
- Quand un widget apparaît, interagis avec lui AVANT de taper un message.
- Envoie des messages naturels, pas des commandes robotiques.
- Si l'assistant demande quelque chose, réponds dans ton personnage.
- Essaie d'atteindre la phase finale, mais ne force pas si le chatbot va dans une autre direction.

À LA FIN (OBLIGATOIRE):
1. Quand tu as terminé toutes les phases (ou après avoir atteint la limite de pas),
   cherche le lien "Cliquez ici pour nous aider" tout en bas du chat.
2. Clique dessus. Une popup s'ouvre.
3. Dans la popup, écris un résumé de ton expérience en tant que {persona_name}:
   ce qui t'a plu, ce qui t'a frustré, et une note sur 10 de l'expérience globale.
4. Envoie le formulaire.
5. Ensuite, extrais un résumé de ce qui s'est passé pendant la conversation.
"""
    else:
        footer = """
PACING RULES:
- Wait for the typing indicator (3 dots) to disappear before sending.
- When a widget appears, interact with it BEFORE typing a message.
- Send natural messages, not robotic commands.
- If the assistant asks something, answer in character.
- Try to reach the final phase, but don't force it if the chatbot goes in another direction.

AT THE END (MANDATORY):
1. When you have completed all phases (or reached the step limit),
   look for the link "Cliquez ici pour nous aider" at the very bottom of the chat.
2. Click it. A popup will open.
3. In the popup, write a summary of your experience as {persona_name}:
   what you liked, what frustrated you, and an overall rating out of 10.
4. Submit the form.
5. Then extract a summary of what happened during the conversation.
"""
    return header + "".join(phases) + footer.format(persona_name=persona.name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_task_prompt(persona: PersonaDefinition, yaml_config: dict) -> str:
    """Build the complete task prompt for a browser-use agent.

    Combines three layers:
    1. Site Knowledge — how to navigate travliaq.com
    2. Persona Instructions — who the agent is
    3. Conversation Flow — what to do, phase by phase
    """
    planner_url = yaml_config["target"]["planner_url_clean"]

    # Layer 1
    site_template = SITE_KNOWLEDGE_FR if persona.language == "fr" else SITE_KNOWLEDGE_EN
    site_section = site_template.format(planner_url=planner_url)

    # Layer 2
    persona_section = _build_persona_section(persona)

    # Layer 3
    flow_section = _build_flow_section(persona)

    # Navigation instruction
    first_goal = persona.conversation_goals[0] if persona.conversation_goals else None
    if first_goal and first_goal.example_message:
        nav_instruction = (
            f"\nPREMIÈRE ACTION: Navigue vers {planner_url}\n"
            f'Ensuite, tape le message suivant dans le champ de texte et envoie-le:\n'
            f'"{first_goal.example_message}"\n'
            if persona.language == "fr"
            else f"\nFIRST ACTION: Navigate to {planner_url}\n"
            f'Then type the following message in the text input and send it:\n'
            f'"{first_goal.example_message}"\n'
        )
    else:
        nav_instruction = (
            f"\nPREMIÈRE ACTION: Navigue vers {planner_url}\n"
            if persona.language == "fr"
            else f"\nFIRST ACTION: Navigate to {planner_url}\n"
        )

    # Final reminder — placed LAST for maximum recency effect
    if persona.language == "fr":
        final_reminder = f"""
##############################################
# RAPPEL FINAL — ACTION OBLIGATOIRE         #
##############################################

AVANT de terminer, tu DOIS accomplir cette dernière action :

1. Scroll tout en bas du panneau de chat.
2. Trouve le lien "Cliquez ici pour nous aider" situé SOUS le champ de texte.
3. CLIQUE sur ce lien. Une popup va s'ouvrir.
4. Dans la popup, écris un résumé de ton expérience en tant que {persona.name} :
   - Ce qui t'a plu
   - Ce qui t'a frustré
   - Une note sur 10
5. Clique sur le bouton d'envoi dans la popup.

⚠️ Si tu ne fais pas cette action, le test est considéré comme ÉCHOUÉ.
"""
    else:
        final_reminder = f"""
##############################################
# FINAL REMINDER — MANDATORY ACTION         #
##############################################

BEFORE finishing, you MUST complete this final action:

1. Scroll to the very bottom of the chat panel.
2. Find the link "Cliquez ici pour nous aider" located BELOW the text input.
3. CLICK this link. A popup will open.
4. In the popup, write a summary of your experience as {persona.name}:
   - What you liked
   - What frustrated you
   - A rating out of 10
5. Click the submit button in the popup.

⚠️ If you do NOT complete this action, the test is considered FAILED.
"""

    return site_section + persona_section + flow_section + nav_instruction + final_reminder
