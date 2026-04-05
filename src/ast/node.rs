#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Id(pub usize);

#[derive(Debug, Clone)]
pub enum ENode {
    Sink {
        input_anchor: super::AnchorId,
    },
    FunctionCall {
        function_declaration_id: super::FunctionDeclarationId,
        input_anchors: Vec<super::AnchorId>,
        output_anchor: super::AnchorId,
    },
    TypeIntroduction {
        r#type: EType,
        output_anchor: super::AnchorId,
    },
    TypeElimination {
        r#type: EType,
        input_anchor: super::AnchorId,
        output_anchor: super::AnchorId,
    },
    Match {
        input_anchor: super::AnchorId,
        output_anchor: super::AnchorId,
    },
}

#[derive(Debug, Clone)]
pub enum EType {
    Bool { value: Option<String> },
    Int { value: Option<String> },
    Float { value: Option<String> },
    String { value: Option<String> },
    Char { value: Option<String> },
    Any,
    Undefined,
    Exception { message: Option<String> },
}

impl ENode {
    pub fn label(
        &self,
        function_declarations: &std::collections::HashMap<
            super::FunctionDeclarationId,
            super::FunctionDeclaration,
        >,
    ) -> String {
        match self {
            ENode::Sink { .. } => "sink".to_string(),
            ENode::FunctionCall {
                function_declaration_id,
                ..
            } => function_declarations
                .get(&function_declaration_id)
                .unwrap()
                .name
                .to_string(),
            ENode::TypeIntroduction { r#type, .. } | ENode::TypeElimination { r#type, .. } => {
                r#type.to_string()
            }
            ENode::Match { .. } => "match true".to_string(),
        }
    }

    pub fn anchors(&self) -> Vec<(super::AnchorId, super::EAnchor)> {
        match self {
            ENode::FunctionCall {
                input_anchors,
                output_anchor,
                ..
            } => input_anchors
                .clone()
                .into_iter()
                .enumerate()
                .map(|(i, anchor_id)| {
                    (
                        anchor_id,
                        super::EAnchor::Input {
                            order_num: i,
                            name: Some(format!("param{}", i)),
                        },
                    )
                })
                .chain(vec![(output_anchor.clone(), super::EAnchor::Output)])
                .collect(),
            ENode::Sink { input_anchor } => vec![(
                input_anchor.clone(),
                super::EAnchor::Input {
                    order_num: 0,
                    name: None,
                },
            )],
            ENode::TypeIntroduction { output_anchor, .. } => {
                vec![(output_anchor.clone(), super::EAnchor::Output)]
            }
            ENode::TypeElimination {
                input_anchor,
                output_anchor,
                ..
            } => vec![
                (
                    input_anchor.clone(),
                    super::EAnchor::Input {
                        order_num: 0,
                        name: None,
                    },
                ),
                (output_anchor.clone(), super::EAnchor::Output),
            ],
            ENode::Match {
                input_anchor,
                output_anchor,
            } => vec![
                (
                    input_anchor.clone(),
                    super::EAnchor::Input {
                        order_num: 0,
                        name: None,
                    },
                ),
                (output_anchor.clone(), super::EAnchor::Output),
            ],
        }
    }
}

impl ToString for EType {
    fn to_string(&self) -> String {
        match self.clone() {
            EType::Bool { value } => value.unwrap_or("bool".to_string()),
            EType::Int { value } => value.unwrap_or("int".to_string()),
            EType::Float { value } => value.unwrap_or("float".to_string()),
            EType::String { value } => {
                if let Some(value) = value {
                    format!("\"{}\"", value)
                } else {
                    "string".to_string()
                }
            }
            EType::Char { value } => {
                if let Some(value) = value {
                    format!("'{}'", value)
                } else {
                    "char".to_string()
                }
            }
            EType::Any => "any".to_string(),
            EType::Undefined => "undefined".to_string(),
            EType::Exception { message } => {
                if let Some(message) = message {
                    format!("exception({})", message)
                } else {
                    "exception".to_string()
                }
            }
        }
    }
}
