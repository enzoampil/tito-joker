until streamlit run tito-jokes-app.py; do
            echo "Tito Joker crashed with exit code $?.  Respawning.." >&2
                sleep 1
        done
