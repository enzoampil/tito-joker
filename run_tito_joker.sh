until streamlit run tito_joker_app.py; do
            echo "Tito Joker crashed with exit code $?.  Respawning.." >&2
                sleep 1
        done
