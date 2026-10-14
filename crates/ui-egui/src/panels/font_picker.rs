//! The font picker dropdown shared by every font field: search-as-you-type, starred
//! favorites, recent picks, then all families. Every row renders in its own typeface where the
//! interface fonts cover it (bundled, craft-fonts and the installed CJK / right-to-left faces);
//! other families fall back to the interface face. Favorites and recents live in the graphics
//! preferences, so every picker shares them.
//!
//! Automation: `${auto_id}` (the combo) and `${auto_id}.option.<family>` (visible rows).

use egui::{FontFamily, FontId, RichText, Ui};

use crate::FilmcraftApp;
use crate::theme::Tokens;

/// Recently picked families kept per list.
pub const RECENT_CAP: usize = 8;

/// One picker section (headers are translated at the call site with `tl!` literals).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Favorites,
    Recent,
    All,
}

/// Sectioned, filtered family names. Pure, so tests cover the ordering.
pub fn sections(
    families: &[(String, Vec<String>)],
    favorites: &[String],
    recent: &[String],
    query: &str,
) -> Vec<(Section, Vec<String>)> {
    let q = query.to_lowercase();
    let matches = |f: &str| q.is_empty() || f.to_lowercase().contains(&q);
    let known: Vec<&str> = families.iter().map(|(f, _)| f.as_str()).collect();
    let in_list = |list: &[String]| -> Vec<String> {
        list.iter()
            .filter(|f| known.iter().any(|k| k.eq_ignore_ascii_case(f)))
            .filter(|f| matches(f))
            .map(|f| known.iter().find(|k| k.eq_ignore_ascii_case(f)).map(|k| k.to_string()).unwrap_or_else(|| f.clone()))
            .collect()
    };
    let all: Vec<String> = families.iter().map(|(f, _)| f.clone()).filter(|f| matches(f)).collect();
    vec![(Section::Favorites, in_list(favorites)), (Section::Recent, in_list(recent)), (Section::All, all)]
}

/// Remember a pick in the recents (persisted with the preferences).
fn remember(app: &mut FilmcraftApp, family: &str) {
    let mut prefs = app.session.prefs.clone();
    prefs.graphics.font_recent.retain(|f| !f.eq_ignore_ascii_case(family));
    prefs.graphics.font_recent.insert(0, family.to_string());
    prefs.graphics.font_recent.truncate(RECENT_CAP);
    if prefs != app.session.prefs {
        if let Err(e) = app.session.set_prefs(prefs) {
            app.ui.status = format!("saving preferences: {e}");
        }
    }
}

/// Star or unstar a family (persisted with the preferences).
fn toggle_favorite(app: &mut FilmcraftApp, family: &str) {
    let mut prefs = app.session.prefs.clone();
    if prefs.graphics.font_favorites.iter().any(|f| f.eq_ignore_ascii_case(family)) {
        prefs.graphics.font_favorites.retain(|f| !f.eq_ignore_ascii_case(family));
    } else {
        prefs.graphics.font_favorites.insert(0, family.to_string());
    }
    if prefs != app.session.prefs {
        if let Err(e) = app.session.set_prefs(prefs) {
            app.ui.status = format!("saving preferences: {e}");
        }
    }
}

fn section_head(ui: &mut Ui, text: &str, t: Tokens) {
    ui.label(RichText::new(text).size(11.0).color(t.text_dim).strong());
}

/// A font dropdown for the family `current`. Returns the picked family (if any) and the combo
/// response (for automation ids). Typing filters; Enter picks the exact match, else the first
/// filtered family.
pub fn show(app: &mut FilmcraftApp, ui: &mut Ui, id_salt: &str, current: &str, width: f32, auto_id: &str) -> (Option<String>, egui::Response) {
    if !filmcraft_text::fonts::system_scanned() {
        filmcraft_text::fonts::scan_system();
    }
    let mut families = filmcraft_text::families();
    // names starting with '.' are macOS-private system faces (UI/fallback fonts, some without
    // a space glyph)
    families.retain(|(f, _)| !f.starts_with('.'));
    let t = app.tokens;
    // families bound to interface fonts: only these can preview rows in their own
    // typeface (an unbound FontFamily::Name panics inside epaint instead of falling back)
    let registered: std::collections::HashSet<String> = ui.ctx().fonts_mut(|f| {
        f.definitions()
            .families
            .keys()
            .filter_map(|fam| match fam {
                FontFamily::Name(n) => Some(n.to_string()),
                _ => None,
            })
            .collect()
    });
    let search_id = (id_salt.to_string(), "font-search");
    let search_id = egui::Id::new(search_id);
    let mut picked: Option<String> = None;
    let combo = egui::ComboBox::from_id_salt((id_salt.to_string(), "combo"))
        .selected_text(current)
        .width(width)
        .height(420.0)
        .show_ui(ui, |ui| {
            let mut query = ui.data(|d| d.get_temp::<String>(search_id)).unwrap_or_default();
            let search = ui.add(egui::TextEdit::singleline(&mut query).hint_text(tl!("Search")).desired_width(ui.available_width()));
            if search.changed() {
                ui.data_mut(|d| d.insert_temp(search_id, query.clone()));
            }
            let enter = search.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            egui::ScrollArea::vertical().max_height(360.0).auto_shrink([false, false]).show(ui, |ui| {
                for (section, names) in sections(&families, &app.session.prefs.graphics.font_favorites, &app.session.prefs.graphics.font_recent, &query) {
                    if names.is_empty() {
                        continue;
                    }
                    section_head(
                        ui,
                        match section {
                            Section::Favorites => tl!("Favorites"),
                            Section::Recent => tl!("Recent"),
                            Section::All => tl!("All"),
                        },
                        t,
                    );
                    for fam in &names {
                        let starred = app.session.prefs.graphics.font_favorites.iter().any(|f| f.eq_ignore_ascii_case(fam));
                        ui.horizontal(|ui| {
                            let star = ui.selectable_label(starred, if starred { "★" } else { "☆" });
                            if star.clicked() {
                                toggle_favorite(app, fam);
                            }
                            // preview in its own typeface when bound; otherwise the plain name
                            // with a ✕ whose hover note says the preview failed
                            let face = if registered.contains(fam) {
                                RichText::new(fam).font(FontId::new(12.0, FontFamily::Name(fam.as_str().into()))).size(12.0)
                            } else {
                                RichText::new(fam).size(12.0)
                            };
                            let row = ui.selectable_label(false, face);
                            let visible = row.rect.intersect(ui.clip_rect());
                            if visible.is_positive() {
                                app.auto.add(&format!("{auto_id}.option.{fam}"), visible, fam);
                            }
                            if row.clicked() {
                                picked = Some(fam.clone());
                            }
                            if !registered.contains(fam) {
                                let _ = ui.label("✕").on_hover_text(tl!("Preview unavailable for this font"));
                            }
                        });
                    }
                }
            });
            if enter {
                let q = query.trim().to_lowercase();
                if !q.is_empty() {
                    let all: Vec<String> = families.iter().map(|(f, _)| f.clone()).collect();
                    picked = all.iter().find(|f| f.to_lowercase() == q).or_else(|| all.iter().find(|f| f.to_lowercase().contains(&q))).cloned();
                }
            }
        });
    if let Some(fam) = picked.clone() {
        remember(app, &fam);
    }
    app.auto.add(auto_id, combo.response.rect, current);
    (picked, combo.response)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fams() -> Vec<(String, Vec<String>)> {
        vec![("Inter".into(), vec!["Regular".into()]), ("Arial".into(), vec!["Regular".into()]), ("Noto Sans Arabic".into(), vec!["Regular".into()])]
    }

    #[test]
    fn sections_order_filter_and_case() {
        let all = sections(&fams(), &["arial".into()], &["Inter".into()], "");
        assert_eq!(all[0], (Section::Favorites, vec!["Arial".to_string()]), "stored case maps to the real name");
        assert_eq!(all[1].1, vec!["Inter".to_string()]);
        assert_eq!(all[2].1.len(), 3);
        let filtered = sections(&fams(), &[], &[], "arab");
        assert!(filtered[0].1.is_empty() && filtered[1].1.is_empty());
        assert_eq!(filtered[2].1, vec!["Noto Sans Arabic".to_string()]);
        let missing = sections(&fams(), &["Gone".into()], &[], "");
        assert!(missing[0].1.is_empty(), "uninstalled favorites hide");
    }
}
