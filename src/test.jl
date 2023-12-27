using Flux
using Transformers
using Transformers.HuggingFace

# THE TEXT ----
synopsis::String = "Путешественник останавливается в старинном замке, где наблюдает ужасные видения: человека, тень которого живет самостоятельной жизнью, странную фигуру с косой, сцену собственной смерти. Затем события принимают ещё более зловещий оборот."

genres::Vector{Symbol} = [
    :ужасы,
    :фэнтези,
    :комедия,
    :мелодрама,
    :драма,
    :боевик,
    :криминал,
    :детектив
]

# THE MODEL ----
model_name = "cointegrated/rubert-base-cased-nli-twoway"
tkr = load_tokenizer(model_name)

HuggingFace.get_model_type(:bert)

initial_cfg = load_config(model_name)
initial_cfg.num_labels
genre_cfg = HuggingFace.HGFConfig(initial_cfg; num_labels=length(genres))
genre_cfg.num_labels
model = load_model(
    model_name;
    task="forsequenceclassification",
    config=genre_cfg,
    trainmode=false, local_files_only=false, cache=true
)
HuggingFace.get_state_dict(model)

# TOKENIZE ----
typeof(model)
tokens = Transformers.encode(tkr, synopsis, String.(genres))
new_model = model(tokens)